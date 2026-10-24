"""Build the requested experiment locally. Does not upload or start GPU sessions."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from prepare_kernel import package

from datasets import load_from_disk
from src.model import CodeTransformer
from src.trainer.code_trainer import make_batch_sampler, training_plan
from src.utils.code_runtime import model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="kaggle/build/pre768-unlimited")
    parser.add_argument("--prepared-dir", default="../data/prepared/code8k")
    parser.add_argument("--owner", default="andrewkhoroshilov")
    parser.add_argument("--code-slug", default="nlp-hw2-code-pre768")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume-training", help="Exact owner/kernel/version providing last.pt"
    )
    resume.add_argument(
        "--resume-dataset", help="Own private owner/dataset with last.pt and best.pt"
    )
    parser.add_argument("--comet-workspace")
    parser.add_argument("--comet-project", default="NLP_HW2")
    parser.add_argument(
        "--inference-plan", default="src/configs/inference_sweeps/pre768.json"
    )
    args = parser.parse_args()
    inference_plan = Path(args.inference_plan).resolve()
    if not inference_plan.is_relative_to(PROJECT) or not inference_plan.is_file():
        parser.error("--inference-plan must be an existing file inside the project")
    from src.utils.inference_sweep import validate_plan

    validate_plan(json.loads(inference_plan.read_text()))
    plan_relative = str(inference_plan.relative_to(PROJECT))
    prepared = Path(args.prepared_dir).resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    overrides = ["experiments=pre768_3epochs", "writer.mode=disabled"]
    if args.comet_workspace:
        overrides.append(f"writer.workspace={args.comet_workspace}")
    overrides.append(f"writer.project_name={args.comet_project}")
    with initialize_config_dir(str(PROJECT / "src/configs"), version_base=None):
        config = compose(config_name="code_baseline", overrides=overrides)
    architecture = model_config(config, manifest)
    sampler = make_batch_sampler(
        load_from_disk(str(prepared / "train")),
        config.dataloader,
        rank=0,
        world_size=2,
        seed=config.seed,
    )
    plan = training_plan(config.trainer, sampler, config.trainer.accumulation_steps)
    output = package(
        args.output,
        args.owner,
        code_slug=args.code_slug,
        kernel_slug="nlp-hw2-pre768-train",
        overrides=overrides,
        max_hours=None,
    )
    OmegaConf.save(config, output / "requested-config.yaml", resolve=True)
    (output / "requested-config.json").write_text(
        json.dumps(OmegaConf.to_container(config, resolve=True), indent=2)
    )
    (output / "expected-training-plan.json").write_text(json.dumps(plan, indent=2))
    settings = {
        "overrides": overrides,
        "max_hours": None,
        "preflight_model": architecture,
        "preflight_script": (PROJECT / "kaggle/model_preflight.py").read_text(),
    }
    if args.resume_training or args.resume_dataset:
        settings["resume_training"] = (
            args.resume_training or f"dataset:{args.resume_dataset}"
        )
        settings["resume_run_name"] = config.run_name
        import zipfile

        with zipfile.ZipFile(output / "code_dataset/code_bundle.zip") as archive:
            settings["expected_source_manifest"] = json.loads(
                archive.read("bundle_manifest.json")
            )
        settings["preflight_script"] = None
    (output / "kernel/run.py").write_text(
        "SETTINGS = "
        + repr(settings)
        + "\n\n"
        + (PROJECT / "kaggle/train_launcher.py").read_text()
    )
    metadata_path = output / "kernel/kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["title"] = "NLP HW2 pre768 train"
    if args.resume_training:
        metadata["kernel_sources"] = [args.resume_training]
    if args.resume_dataset:
        metadata["dataset_sources"].append(args.resume_dataset)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    dataset_path = output / "code_dataset/dataset-metadata.json"
    metadata = json.loads(dataset_path.read_text())
    metadata["title"] = "NLP HW2 pre768 code bundle"
    dataset_path.write_text(json.dumps(metadata, indent=2))
    inference = output / "inference"
    inference.mkdir()
    (inference / "run.py").write_text(
        "SETTINGS = "
        + repr({"sweep_plan": plan_relative})
        + "\n\n"
        + (PROJECT / "kaggle/inference_launcher.py").read_text()
    )
    (inference / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{args.owner}/nlp-hw2-pre768-inference",
                "title": "NLP HW2 pre768 inference",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "machine_shape": "NvidiaTeslaT4",
                "enable_internet": True,
                "dataset_sources": [f"{args.owner}/nlp-hw2-data"],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
            },
            indent=2,
        )
    )
    import torch

    with torch.device("meta"):
        model = CodeTransformer(**architecture)
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "status": "prepared_not_launched",
                "parameters": sum(p.numel() for p in model.parameters()),
                "max_epochs": 3,
                "max_job_hours": None,
                "world_size": 2,
                "manifest_sha256": hashlib.sha256(
                    (prepared / "manifest.json").read_bytes()
                ).hexdigest(),
                "tokenizer_sha256": manifest["tokenizer_sha256"],
                "inference_source_binding": "Pinned to returned training version by launch_prepared.py",
            },
            indent=2,
        )
    )
    print(output)
    print("Prepared only; no upload or GPU training was started.")


if __name__ == "__main__":
    main()
