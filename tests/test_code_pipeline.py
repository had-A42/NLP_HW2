import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from average_checkpoints import average
from src.datasets.code_collate import CodeCollator
from src.datasets.code_preparation import is_validation
from src.datasets.code_sampler import CodeBatchSampler
from src.model import CodeTransformer
from src.tokenization import CodeTokenizer
from src.utils.code_runtime import load_checkpoint, save_checkpoint


def tiny_model(**kwargs):
    return CodeTransformer(
        300,
        d_model=16,
        n_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        ffn_dim=32,
        dropout=0,
        **kwargs,
    )


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_incremental_logits_match_full_prefix(self):
        for options in [
            {},
            {"norm_style": "pre"},
            {"norm_type": "rms", "activation": "swiglu"},
            {"position_encoding": "rope", "kv_heads": 1, "tie_output": True},
            {"layer_repeats": 3},
            {
                "layer_repeats": 3,
                "norm_style": "pre",
                "position_encoding": "rope",
                "kv_heads": 1,
            },
        ]:
            with self.subTest(options=options), torch.inference_mode():
                torch.manual_seed(42)
                model = tiny_model(**options).eval()
                source = torch.tensor([[4, 5, 2, 0], [6, 7, 8, 2]])
                decoder = torch.tensor([[1, 10, 11], [1, 12, 13]])
                memory, allowed = model.encode(source)
                full = model.decode(decoder, memory, allowed)
                for kind in ["static", "dynamic"]:
                    states = model.init_cache(memory, 4, kind)
                    for position in range(3):
                        output = model.decode_step(
                            decoder[:, position],
                            states,
                            torch.tensor(position),
                            allowed,
                            kind,
                        )
                        torch.testing.assert_close(
                            output, full[:, position], atol=1e-5, rtol=1e-4
                        )

    def test_causality_and_source_padding(self):
        model = tiny_model(layer_repeats=3).eval()
        source = torch.tensor([[4, 5, 2, 0]])
        decoder = torch.tensor([[1, 10, 11, 12]])
        with torch.inference_mode():
            original = model(source, decoder)["logits"]
            changed = decoder.clone()
            changed[:, 2:] = torch.tensor([50, 51])
            torch.testing.assert_close(
                original[:, :2], model(source, changed)["logits"][:, :2]
            )
            mask = source.ne(0)
            changed_source = source.clone()
            changed_source[:, -1] = 55
            torch.testing.assert_close(
                original, model(changed_source, decoder, mask)["logits"]
            )

    def test_byte_bpe_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            texts = ["def solve():\n    return 42\n", '  x = "<eos>"\n\t# тест é 🙂\n']
            tokenizer = CodeTokenizer.train(
                iter(texts), Path(temp) / "tokenizer.json", 300, 1
            )
            for text in texts:
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_layer_repeats_share_weights_and_repeat_each_layer_consecutively(self):
        torch.manual_seed(42)
        reference = CodeTransformer(
            300,
            d_model=16,
            n_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=32,
            dropout=0,
        )
        torch.manual_seed(42)
        model = CodeTransformer(
            300,
            d_model=16,
            n_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=32,
            dropout=0,
            layer_repeats=3,
        )
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in reference.parameters()),
        )
        for key, weights in reference.state_dict().items():
            torch.testing.assert_close(weights, model.state_dict()[key], rtol=0, atol=0)
        order = []
        hooks = []
        for side, layers in [("encoder", model.encoder), ("decoder", model.decoder)]:
            for i, layer in enumerate(layers):
                hooks.append(
                    layer.register_forward_hook(
                        lambda module, args, result, label=f"{side}{i}": order.append(
                            label
                        )
                    )
                )
        logits = model(torch.tensor([[4, 5, 2]]), torch.tensor([[1, 6, 7]]))["logits"]
        self.assertEqual(
            order,
            ["encoder0"] * 3 + ["encoder1"] * 3 + ["decoder0"] * 3 + ["decoder1"] * 3,
        )
        logits.square().mean().backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        for hook in hooks:
            hook.remove()
        model.eval()
        with torch.inference_mode():
            memory, allowed = model.encode(torch.tensor([[4, 5, 2]]))
            states = model.init_cache(memory, 4)
            self.assertEqual(len(states), 6)
            self.assertEqual(len({s["k"].data_ptr() for s in states}), 6)
            self.assertEqual(len({s["v"].data_ptr() for s in states}), 6)
            with self.assertRaisesRegex(ValueError, "one state per layer execution"):
                model.decode_step(
                    torch.tensor([1]), states[:2], torch.tensor(0), allowed
                )
            for kind in ["static", "dynamic"]:
                result = model.generate(
                    torch.tensor([[4, 5, 2]]),
                    max_new_tokens=4,
                    num_beams=2,
                    cache_type=kind,
                )
                self.assertEqual(result.shape[0], 1)

    def test_invalid_layer_repeats(self):
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                tiny_model(layer_repeats=value)

    def test_collator_shift_and_ignored_padding(self):
        batch = CodeCollator()(
            [
                {"row_id": 4, "source_ids": [8, 2], "target_ids": [9, 10, 2]},
                {"row_id": 5, "source_ids": [7, 8, 2], "target_ids": [11, 2]},
            ]
        )
        self.assertEqual(batch["decoder_input_ids"].tolist(), [[1, 9, 10], [1, 11, 2]])
        self.assertEqual(batch["labels"].tolist(), [[9, 10, 2], [11, 2, -100]])
        self.assertEqual(
            batch["attention_mask"].tolist(), [[True, True, False], [True, True, True]]
        )

    def test_group_partition_is_stable(self):
        questions = [f"question {i}" for i in range(1000)]
        assignments = {q: is_validation(q, 0.1, 42) for q in questions}
        self.assertGreater(sum(assignments.values()), 0)
        self.assertLess(sum(assignments.values()), len(questions))
        for q in reversed(questions):
            self.assertEqual(assignments[q], is_validation(q, 0.1, 42))

    def test_sampler_disjointness_replay_and_resume(self):
        lengths = np.arange(40) % 11
        for bucketed in [False, True]:
            samplers = [
                CodeBatchSampler(lengths, 2, rank=rank, world_size=2, bucketed=bucketed)
                for rank in range(2)
            ]
            batches = [list(s) for s in samplers]
            flattened = [
                idx
                for rank_batches in batches
                for batch in rank_batches
                for idx in batch
            ]
            self.assertEqual(sorted(flattened), list(range(40)))
            self.assertEqual(batches[0], list(samplers[0]))
            samplers[0].set_epoch(0, 3)
            self.assertEqual(list(samplers[0]), batches[0][3:])

    def test_generation_eos_and_beam_batches(self):
        class EOSModel(CodeTransformer):
            def decode_step(
                self, last_token, states, position, allowed, cache_type="static"
            ):
                logits = torch.full(
                    (len(last_token), 300), -10.0, device=last_token.device
                )
                logits[:, self.eos_id] = 10
                return logits

        model = EOSModel(
            300,
            d_model=16,
            n_heads=2,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=32,
            dropout=0,
        ).eval()
        source = torch.tensor([[4, 2], [5, 2]])
        result = model.generate(source, max_new_tokens=4)
        self.assertEqual(result.tolist(), [[2], [2]])
        result = model.generate(source, max_new_tokens=4, num_beams=2)
        self.assertEqual(result[:, 0].tolist(), [2, 2])

    def test_pre_norm_beam_matches_exhaustive_tiny_search(self):
        # Three allowed continuations, depth two: all seven complete/truncated
        # sequences fit in beam=8. Compute their scores without using KV caches.
        torch.manual_seed(17)
        model = tiny_model(norm_style="pre").eval()
        source = torch.tensor([[4, 2, 0], [5, 6, 2]])
        suppress = tuple(i for i in range(300) if i not in {2, 3, 4})
        with torch.inference_mode():
            expected = []
            for row in source:
                memory, mask = model.encode(row[None])
                candidates = []

                def explore(prefix, score):
                    if (prefix and prefix[-1] == 2) or len(prefix) == 2:
                        candidates.append(
                            (score / ((5 + len(prefix)) / 6) ** 0.6, prefix)
                        )
                        return
                    logits = model.decode(torch.tensor([[1] + prefix]), memory, mask)[
                        0, -1
                    ]
                    probs = logits[torch.tensor([2, 3, 4])].log_softmax(0)
                    for token, logp in zip([2, 3, 4], probs):
                        explore(prefix + [token], score + logp.item())

                explore([], 0.0)
                expected.append(max(candidates, key=lambda item: item[0])[1])
            for kind in ["static", "dynamic"]:
                output = model.generate(
                    source,
                    max_new_tokens=2,
                    num_beams=8,
                    cache_type=kind,
                    suppress_tokens=suppress,
                )
                actual = []
                for row in output.tolist():
                    actual.append(row[: row.index(2) + 1] if 2 in row else row)
                self.assertEqual(actual, expected)

    def test_checkpoint_averaging_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = {
                "format_version": 1,
                "model_config": {"vocab_size": 300},
                "tokenizer_sha256": "tokenizer",
                "manifest_sha256": "manifest",
                "config": {},
            }
            for i in range(2):
                save_checkpoint(
                    root / f"{i}.pt",
                    {**metadata, "state_dict": {"weight": torch.tensor([float(i)])}},
                )
            output = average([root / "0.pt", root / "1.pt"], root / "average.pt")
            self.assertEqual(
                load_checkpoint(output)["state_dict"]["weight"].item(), 0.5
            )
            bad = {
                **metadata,
                "tokenizer_sha256": "different",
                "state_dict": {"weight": torch.tensor([1.0])},
            }
            save_checkpoint(root / "bad.pt", bad)
            with self.assertRaises(ValueError):
                average([root / "0.pt", root / "bad.pt"], root / "bad-average.pt")

    def test_averaging_tied_storage_does_not_double_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = {
                "format_version": 1,
                "model_config": {},
                "config": {},
                "tokenizer_sha256": "same",
                "manifest_sha256": "same",
            }
            for i in range(2):
                shared = torch.tensor([float(i)])
                save_checkpoint(
                    root / f"{i}.pt",
                    {**metadata, "state_dict": {"embedding": shared, "output": shared}},
                )
            output = average([root / "0.pt", root / "1.pt"], root / "averaged.pt")
            state = load_checkpoint(output)["state_dict"]
            self.assertEqual(state["embedding"].item(), 0.5)
            self.assertEqual(state["output"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
