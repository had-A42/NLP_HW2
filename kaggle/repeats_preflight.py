"""Single-T4 safety profile before the shared-layer-repeat training experiment.

Uses randomly initialized weights. No generated programs are saved or used as
training targets; EOS is suppressed to measure the full 128-step decode cost.
"""

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
    root, manifest, tokenizer = prepared_artifacts(os.environ["HW2_PREPARED_DIR"])
    model = (
        CodeTransformer(
            vocab_size=manifest["vocab_size"],
            layer_repeats=3,
            max_source_length=manifest["max_source_length"],
            max_target_length=manifest["max_target_length"],
        )
        .cuda()
        .train()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.98), fused=True
    )
    scaler = torch.amp.GradScaler("cuda")
    # Largest allowed source/target lengths with exactly 16,384 padded tokens.
    train = load_from_disk(str(root / "train"))
    batch = CodeCollator(tokenizer.pad_id, tokenizer.bos_id)(
        [train[i] for i in range(32)]
    )
    source = F.pad(
        batch["input_ids"],
        (0, 256 - batch["input_ids"].shape[1]),
        value=tokenizer.pad_id,
    ).cuda()
    decoder = F.pad(
        batch["decoder_input_ids"],
        (0, 256 - batch["decoder_input_ids"].shape[1]),
        value=tokenizer.pad_id,
    ).cuda()
    labels = F.pad(
        batch["labels"], (0, 256 - batch["labels"].shape[1]), value=-100
    ).cuda()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(source, decoder)["logits"]
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite repeated-layer training loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss))
    peak = torch.cuda.max_memory_allocated() / 2**30
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler, logits, loss, source, decoder, labels
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
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(rows), 128):
        batch = collate(rows[start : start + 128])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            model.generate(
                batch["input_ids"].cuda(),
                batch["attention_mask"].cuda(),
                max_new_tokens=128,
                suppress_tokens=(tokenizer.eos_id, tokenizer.unk_id),
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = dict(
        technical_preflight=True,
        trained_checkpoint=False,
        parameters=sum(p.numel() for p in model.parameters()),
        layer_repeats=3,
        worst_training_shape=[32, 256, 256],
        padded_tokens=16384,
        peak_training_allocated_gib=peak,
        finite_losses=losses,
        test_rows=len(rows),
        decode_batch_size=128,
        forced_decode_steps=128,
        warm_decode_seconds=elapsed,
        visible_cuda_devices=torch.cuda.device_count(),
    )
    Path("/kaggle/working/repeats_preflight.json").write_text(
        json.dumps(report, indent=2)
    )
    print("REPEATS_PREFLIGHT " + json.dumps(report), flush=True)
    if elapsed > 160:
        raise RuntimeError(
            "Preflight has insufficient margin for a 180-second cold inference."
        )


if __name__ == "__main__":
    main()
