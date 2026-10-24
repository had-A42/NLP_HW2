"""DDP/AMP training, globally token-weighted accumulation, validation and resume."""

import contextlib
import itertools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from datasets import load_from_disk
from src.datasets.code_collate import CodeCollator
from src.datasets.code_sampler import CodeBatchSampler, TokenBudgetBatchSampler
from src.loss import TokenCrossEntropy
from src.model import CodeTransformer
from src.utils.code_runtime import (
    CodeLogger,
    environment_report,
    load_checkpoint,
    model_config,
    prepared_artifacts,
    save_checkpoint,
    sha256,
)


def learning_rate_factor(step, warmup_steps, total_steps, minimum=0.1):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def wall_time_limit(hours):
    if hours is None:
        return math.inf
    if isinstance(hours, bool) or not math.isfinite(hours) or hours <= 0:
        raise ValueError("trainer.max_hours must be positive or null to disable.")
    return hours * 3600


def training_plan(settings, sampler, accumulation):
    """Resolve an epoch-aware schedule from the actual deterministic batch plans."""
    max_steps = settings.get("max_steps")
    max_epochs = settings.get("max_epochs")
    for name, value in [("max_steps", max_steps), ("max_epochs", max_epochs)]:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"trainer.{name} must be a positive integer or null.")
    if max_steps is None and max_epochs is None:
        raise ValueError("Set trainer.max_steps or trainer.max_epochs.")
    counts = []
    if max_epochs is not None:
        original_epoch, original_offset = sampler.epoch, sampler.start_batch
        try:
            for epoch in range(max_epochs):
                sampler.set_epoch(epoch)
                counts.append(len(sampler))
        finally:
            sampler.set_epoch(original_epoch, original_offset)
        if min(counts) < 1:
            raise ValueError("At least one epoch has no distributed training batches.")
        epoch_steps = sum(math.ceil(count / accumulation) for count in counts)
        max_steps = epoch_steps if max_steps is None else min(max_steps, epoch_steps)
    return {
        "max_epochs": max_epochs,
        "optimizer_step_limit": max_steps,
        "microbatches_per_epoch_per_rank": counts,
        "accumulation_steps": accumulation,
    }


def move_batch(batch, device):
    return {
        k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "row_id"
    }


def make_batch_sampler(data, config, rank, world_size, seed, shuffle=True):
    # Read only length columns and respect logical HF indices, if present.
    lengths = data.select_columns(["source_length", "target_length"]).with_format(
        "numpy"
    )[:]
    mode = config.get("batching", "fixed")
    if mode == "tokens":
        return TokenBudgetBatchSampler(
            lengths["source_length"],
            lengths["target_length"],
            token_budget=int(config.token_budget),
            max_batch_size=int(config.max_batch_size),
            bucket_size=int(config.bucket_size),
            rank=rank,
            world_size=world_size,
            seed=seed,
            shuffle=shuffle,
        )
    if mode == "fixed":
        return CodeBatchSampler(
            lengths["source_length"] + lengths["target_length"],
            int(config.batch_size),
            rank,
            world_size,
            seed,
            bucketed=bool(config.get("bucketed", False)),
            bucket_batches=int(config.get("bucket_batches", 100)),
            shuffle=shuffle,
            drop_last=shuffle,
        )
    raise ValueError(f"Unknown dataloader.batching={mode!r}; use tokens or fixed.")


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def validate(model, loader, criterion, device, amp, max_batches):
    was_training = model.training
    model.eval()
    sums = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.inference_mode():
        for batch in itertools.islice(loader, max_batches if max_batches > 0 else None):
            batch = move_batch(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                losses = criterion(**model(**batch), labels=batch["labels"])
            sums += torch.stack(
                (losses["loss_sum"].double(), losses["token_count"].double())
            )
    model.train(was_training)
    return (
        {"val_loss": (sums[0] / sums[1]).item(), "val_tokens": int(sums[1].item())}
        if sums[1]
        else {}
    )


def run_training(config):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = config.trainer.device
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Code training supports auto, cpu or cuda.")
    cuda = requested != "cpu" and torch.cuda.is_available()
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    torch.set_num_threads(int(config.trainer.cpu_threads))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")
    logger = None
    try:
        seed = int(config.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        root, manifest, tokenizer = prepared_artifacts(config.prepared_dir)
        architecture = model_config(config, manifest)
        manifest_hash = sha256(root / "manifest.json")
        train_data = load_from_disk(str(root / "train"))
        val_data = load_from_disk(str(root / "val"))
        accumulation = int(config.trainer.accumulation_steps)
        if accumulation < 1:
            raise ValueError("accumulation_steps must be positive.")
        time_limit = wall_time_limit(config.trainer.max_hours)
        if (
            min(
                config.trainer.log_every,
                config.trainer.eval_every,
                config.trainer.save_every,
                config.trainer.keep_snapshots,
            )
            < 1
        ):
            raise ValueError(
                "Logging/checkpoint intervals and keep_snapshots must be positive."
            )
        sampler = make_batch_sampler(
            train_data, config.dataloader, rank, world_size, seed
        )
        if not len(sampler):
            raise ValueError(
                "Too few training rows for the configured distributed batches."
            )
        plan = training_plan(config.trainer, sampler, accumulation)
        step_limit, epoch_limit = plan["optimizer_step_limit"], plan["max_epochs"]
        batching_config = {
            "dataloader": OmegaConf.to_container(config.dataloader, resolve=True),
            "seed": seed,
            "world_size": world_size,
            "accumulation_steps": accumulation,
        }
        collate = CodeCollator(tokenizer.pad_id, tokenizer.bos_id)
        loader = DataLoader(
            train_data,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=config.dataloader.num_workers,
            pin_memory=cuda,
            persistent_workers=config.dataloader.num_workers > 0,
            # CUDA/NCCL must not be inherited by forked DataLoader workers.
            multiprocessing_context="spawn" if config.dataloader.num_workers else None,
            timeout=120 if config.dataloader.num_workers else 0,
            generator=torch.Generator().manual_seed(seed + rank),
        )
        val_loader = DataLoader(
            val_data,
            batch_sampler=make_batch_sampler(
                val_data, config.dataloader, 0, 1, seed, shuffle=False
            ),
            collate_fn=collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        model = CodeTransformer(**architecture).to(device)
        amp = bool(cuda and config.trainer.amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        decay, no_decay = [], []
        for parameter in model.parameters():
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.optimizer.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.optimizer.lr,
            betas=tuple(config.optimizer.betas),
            fused=bool(cuda and config.optimizer.fused),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: learning_rate_factor(
                step,
                config.scheduler.warmup_steps,
                step_limit,
                config.scheduler.minimum_ratio,
            ),
        )
        resume = config.trainer.resume_from
        checkpoint = load_checkpoint(resume, "cpu") if resume else None
        step, epoch, offset, tokens_seen, previous_seconds = 0, 0, 0, 0, 0.0
        best_loss = math.inf
        if checkpoint:
            if "optimizer" not in checkpoint:
                raise ValueError(
                    "Resume requires last.pt with optimizer state, not best/averaged/exported weights."
                )
            if (
                checkpoint["model_config"] != architecture
                or checkpoint["manifest_sha256"] != manifest_hash
            ):
                raise ValueError(
                    "Resume checkpoint has incompatible architecture or data/tokenizer."
                )
            if checkpoint.get("batching_config") != batching_config:
                raise ValueError(
                    "Exact resume requires matching batching config, seed and accumulation. "
                    "Checkpoints without batching_config require their original training code."
                )
            if checkpoint.get("training_plan", plan) != plan:
                raise ValueError("Exact resume requires a matching training plan.")
            if checkpoint["world_size"] != world_size:
                raise ValueError("Exact resume requires the same world_size.")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            step, epoch, offset = (
                checkpoint["step"],
                checkpoint["epoch"],
                checkpoint["batch_in_epoch"],
            )
            tokens_seen, best_loss = checkpoint["tokens_seen"], checkpoint["best_loss"]
            previous_seconds = checkpoint["training_seconds"]
        network = (
            DistributedDataParallel(
                model,
                device_ids=[local_rank] if cuda else None,
                broadcast_buffers=False,
            )
            if world_size > 1
            else model
        )
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        if checkpoint:
            restore_rng(checkpoint["rng_states"][rank], device)
        run_dir = Path(config.run_dir).expanduser().resolve()
        if rank == 0:
            if config.writer.mode == "online" and not os.getenv("COMET_API_KEY"):
                raise RuntimeError(
                    "Set COMET_API_KEY or use writer.mode=disabled for local smoke checks."
                )
            if (run_dir / "config.yaml").exists() and not resume:
                raise FileExistsError(
                    f"Run already exists: {run_dir}; choose a new run_name or resume."
                )
            run_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(
                config,
                run_dir / (f"config-resume-{step}.yaml" if resume else "config.yaml"),
                resolve=True,
            )
            (run_dir / "training_plan.json").write_text(json.dumps(plan, indent=2))
            shutil.copyfile(root / "tokenizer.json", run_dir / "tokenizer.json")
            shutil.copyfile(root / "manifest.json", run_dir / "data_manifest.json")
            (run_dir / "environment.json").write_text(
                json.dumps(environment_report(device), indent=2)
            )
            logger = CodeLogger(
                config, run_dir, checkpoint.get("comet_key") if checkpoint else None
            )
            batch_description = (
                f"padded token budget/GPU: {config.dataloader.token_budget}; "
                f"max examples/GPU: {config.dataloader.max_batch_size}"
                if config.dataloader.get("batching", "fixed") == "tokens"
                else f"global batch: {world_size * config.dataloader.batch_size * accumulation}"
            )
            print(
                f"Parameters: {sum(p.numel() for p in model.parameters()):,}; "
                f"{batch_description}; accumulation: {accumulation}; AMP: {amp}",
                flush=True,
            )
            print("TRAINING_PLAN " + json.dumps(plan), flush=True)
        if world_size > 1:
            dist.barrier()
        criterion = TokenCrossEntropy()
        started = time.perf_counter()
        log_started, log_tokens = started, tokens_seen
        skipped = 0

        def save(is_best=False, snapshot=True):
            states = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(states, rng_state(device))
            else:
                states[0] = rng_state(device)
            if rank != 0:
                return
            common = {
                "format_version": 1,
                "state_dict": model.state_dict(),
                "model_config": architecture,
                "step": step,
                "tokenizer_sha256": tokenizer.sha256,
                "manifest_sha256": manifest_hash,
                "batching_config": batching_config,
                "training_plan": plan,
                "config": OmegaConf.to_container(config, resolve=True),
            }
            full = {
                **common,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "batch_in_epoch": offset,
                "tokens_seen": tokens_seen,
                "best_loss": best_loss,
                "world_size": world_size,
                "training_seconds": previous_seconds + time.perf_counter() - started,
                "rng_states": states,
                "comet_key": logger.experiment_key,
            }
            save_checkpoint(run_dir / "last.pt", full)
            if is_best:
                save_checkpoint(run_dir / "best.pt", common)
            if snapshot:
                save_checkpoint(run_dir / f"model-step-{step:08d}.pt", common)
                snapshots = sorted(run_dir.glob("model-step-*.pt"))
                for old in snapshots[: -int(config.trainer.keep_snapshots)]:
                    old.unlink()

        initial_step = step
        stop = (
            step >= step_limit
            or previous_seconds >= time_limit
            or (epoch_limit is not None and epoch >= epoch_limit)
        )
        while not stop:
            sampler.set_epoch(epoch)
            epoch_batches = len(sampler)
            sampler.set_epoch(epoch, offset)
            print(
                f"[rank {rank}] epoch {epoch + 1}: starting DataLoader at batch {offset}/{epoch_batches}",
                flush=True,
            )
            iterator = iter(loader)
            network.train()
            while True:
                window = list(itertools.islice(iterator, accumulation))
                if not window:
                    break
                if step == initial_step:
                    print(f"[rank {rank}] first accumulation window loaded", flush=True)
                # Count the entire accumulation window BEFORE backward. DDP averages
                # gradients, hence each local loss sum is multiplied by world_size.
                window_stats = torch.tensor(
                    [
                        sum(int(batch["labels"].ne(-100).sum()) for batch in window),
                        sum(int(batch["attention_mask"].sum()) for batch in window),
                        sum(
                            batch["input_ids"].numel() + batch["labels"].numel()
                            for batch in window
                        ),
                        sum(len(batch["input_ids"]) for batch in window),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if world_size > 1:
                    dist.all_reduce(window_stats)
                total_count = window_stats[0].float()
                optimizer.zero_grad(set_to_none=True)
                local_loss = torch.zeros((), device=device)
                for micro, cpu_batch in enumerate(window):
                    batch = move_batch(cpu_batch, device)
                    sync = (
                        network.no_sync()
                        if world_size > 1 and micro < len(window) - 1
                        else contextlib.nullcontext()
                    )
                    with sync:
                        with torch.autocast(
                            device.type, dtype=torch.float16, enabled=amp
                        ):
                            losses = criterion(
                                **network(**batch), labels=batch["labels"]
                            )
                            loss = losses["loss_sum"] * world_size / total_count
                        scaler.scale(loss).backward()
                    local_loss += losses["loss_sum"].detach()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.trainer.max_grad_norm
                )
                if not amp and not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite gradient norm.")
                old_scale = scaler.get_scale()
                applied_lr = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                offset += len(window)
                tokens_seen += int(total_count.item())
                overflow = scaler.get_scale() < old_scale
                if overflow:
                    skipped += 1
                    if skipped > 20:
                        raise FloatingPointError(
                            "Repeated FP16 overflow; reduce learning rate or disable AMP."
                        )
                else:
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("Non-finite gradient norm.")
                    skipped = 0
                    scheduler.step()
                    step += 1
                if world_size > 1:
                    dist.all_reduce(local_loss)
                elapsed = previous_seconds + time.perf_counter() - started
                stop_flag = torch.tensor(
                    int(
                        step >= step_limit
                        or elapsed >= time_limit
                        or (
                            epoch_limit is not None
                            and epoch + 1 >= epoch_limit
                            and offset >= epoch_batches
                        )
                    ),
                    device=device,
                )
                if world_size > 1:
                    dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                stop = bool(stop_flag)
                if rank == 0 and (
                    step == initial_step + 1
                    or step % config.trainer.log_every == 0
                    or stop
                ):
                    now = time.perf_counter()
                    metrics = {
                        "train_loss": (local_loss / total_count).item(),
                        "learning_rate": applied_lr,
                        "tokens_seen": tokens_seen,
                        "target_tokens_per_step": int(total_count.item()),
                        "source_tokens_per_step": int(window_stats[1].item()),
                        "padded_tokens_per_step": int(window_stats[2].item()),
                        "examples_per_step": int(window_stats[3].item()),
                        "microbatches_per_step": len(window),
                        "padding_fraction": 1
                        - (
                            (window_stats[0] + window_stats[1]) / window_stats[2]
                        ).item(),
                        "tokens_per_second": (tokens_seen - log_tokens)
                        / max(now - log_started, 1e-9),
                        "grad_norm": float(grad_norm)
                        if torch.isfinite(grad_norm)
                        else 0.0,
                        "training_seconds": elapsed,
                        "epoch": epoch + 1,
                        "batch_in_epoch": offset,
                        "batches_in_epoch": epoch_batches,
                        "epoch_progress_batches": epoch + offset / epoch_batches,
                        "amp_scale": scaler.get_scale(),
                    }
                    if cuda:
                        metrics["peak_gpu_memory"] = torch.cuda.max_memory_allocated(
                            device
                        )
                    logger.log(step, **metrics)
                    log_started, log_tokens = now, tokens_seen
                is_best = False
                if not overflow and (step % config.trainer.eval_every == 0 or stop):
                    values = [
                        validate(
                            model,
                            val_loader,
                            criterion,
                            device,
                            amp,
                            config.trainer.val_max_batches,
                        )
                        if rank == 0
                        else None
                    ]
                    if world_size > 1:
                        dist.broadcast_object_list(values, src=0)
                    metrics = values[0]
                    if metrics:
                        is_best = metrics["val_loss"] < best_loss
                        best_loss = min(best_loss, metrics["val_loss"])
                        if rank == 0:
                            logger.log(step, **metrics)
                if is_best or step % config.trainer.save_every == 0 or stop:
                    save(is_best=is_best)
                if stop:
                    break
            epoch += 1
            offset = 0
            if epoch_limit is not None and epoch >= epoch_limit:
                stop = True
        if checkpoint and step == initial_step and step >= step_limit:
            print("Checkpoint already reached max_steps.", flush=True)
    finally:
        if logger is not None:
            logger.finish()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
