"""Check the requested scratch model before committing to a long GPU training run."""

import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from datasets import load_from_disk
from src.datasets.code_collate import CodeCollator
from src.model import CodeTransformer
from src.utils.code_runtime import prepared_artifacts


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Preflight requires one visible T4.")
    root, _, tokenizer = prepared_artifacts(os.environ["HW2_PREPARED_DIR"])
    model = (
        CodeTransformer(**json.loads(os.environ["HW2_PREFLIGHT_MODEL"])).cuda().train()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.98), fused=True
    )
    scaler = torch.amp.GradScaler("cuda")
    train = load_from_disk(str(root / "train"))
    batch = CodeCollator(tokenizer.pad_id, tokenizer.bos_id)(
        [train[i] for i in range(32)]
    )
    values = {}
    for name, pad in [
        ("input_ids", tokenizer.pad_id),
        ("decoder_input_ids", tokenizer.pad_id),
        ("labels", -100),
    ]:
        values[name] = F.pad(
            batch[name], (0, 256 - batch[name].shape[1]), value=pad
        ).cuda()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(values["input_ids"], values["decoder_input_ids"])["logits"]
            loss = F.cross_entropy(logits.flatten(0, 1), values["labels"].flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite preflight loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite preflight gradient.")
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().item())
    peak = torch.cuda.max_memory_allocated() / 2**30
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler, values, logits, loss
    torch.cuda.empty_cache()
    model.eval()
    test = load_from_disk(str(root / "test"))
    encoded = tokenizer.encode_batch(test["question"])
    rows = sorted(
        [
            dict(row_id=i, source_ids=ids + [tokenizer.eos_id])
            for i, ids in enumerate(encoded)
        ],
        key=lambda row: len(row["source_ids"]),
    )
    collate = CodeCollator(tokenizer.pad_id, tokenizer.bos_id)
    timings = []
    for beams, size in [(1, 128), (2, 64), (4, 32)]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(rows), size):
            batch = collate(rows[start : start + size])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                model.generate(
                    batch["input_ids"].cuda(),
                    batch["attention_mask"].cuda(),
                    max_new_tokens=128,
                    num_beams=beams,
                    suppress_tokens=(tokenizer.eos_id, tokenizer.unk_id),
                )
        torch.cuda.synchronize()
        timings.append(
            dict(
                num_beams=beams,
                batch_size=size,
                forced_steps=128,
                warm_decode_seconds=time.perf_counter() - started,
            )
        )
    report = dict(
        technical_preflight=True,
        trained_checkpoint=False,
        parameters=sum(p.numel() for p in model.parameters()),
        finite_losses=losses,
        padded_training_shape=[32, 256, 256],
        peak_training_allocated_gib=peak,
        test_rows=len(rows),
        visible_cuda_devices=torch.cuda.device_count(),
        timings=timings,
    )
    Path("/kaggle/working/model_preflight.json").write_text(
        json.dumps(report, indent=2)
    )
    print("MODEL_PREFLIGHT " + json.dumps(report), flush=True)
    if timings[0]["warm_decode_seconds"] > 160:
        raise RuntimeError("Insufficient margin for 180-second cold greedy inference.")


if __name__ == "__main__":
    main()
