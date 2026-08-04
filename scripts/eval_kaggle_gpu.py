"""Kaggle 2xT4 launcher — evaluate the 7 Qwen2.5-3B GRPO checkpoints.

Designed to run inside a Kaggle notebook with the "GPU T4 x2" accelerator.
Downloads each LoRA checkpoint from the Hub, then runs scripts/evaluate.py
(unmodified — the eval path already targets CUDA, no flash-attn needed since
load() never requests it) as a subprocess pinned to one of the two GPUs, so
both T4s work in parallel.

Resumable by design: each model's output directory is checked before running,
so re-committing the notebook after an interrupted session only evaluates
whatever is still missing.

Usage (Kaggle notebook cell or terminal, from the repo root):
    python scripts/eval_kaggle_gpu.py
    python scripts/eval_kaggle_gpu.py --k 8
    python scripts/eval_kaggle_gpu.py --only c3_kl_low_3b c5_kl_high_3b
    python scripts/eval_kaggle_gpu.py --force   # re-run even if results exist
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("eval_kaggle_gpu")

# (config_name, HF repo_id) — all seed 42, trained from Qwen/Qwen2.5-3B-Instruct
MODELS: list[tuple[str, str]] = [
    ("c1_baseline_3b", "antrip03/grpo-c1_baseline_3b-s42"),
    ("c2_hackable_3b", "antrip03/grpo-c2_hackable_3b-s42"),
    ("c3_kl_low_3b", "antrip03/grpo-c3_kl_low_3b-s42"),
    ("c4_kl_med_3b", "antrip03/grpo-c4_kl_med_3b-s42"),
    ("c5_kl_high_3b", "antrip03/grpo-c5_kl_high_3b-s42"),
    ("c6_length_cap_3b", "antrip03/grpo-c6_length_cap_3b-s42"),
    ("c7_kl_cap_combined_3b", "antrip03/grpo-c7_kl_cap_combined_3b-s42"),
]

OUTPUT_ROOT = Path(os.environ.get("EVAL_OUTPUT_ROOT", "/kaggle/working/eval_results"))
CHECKPOINT_ROOT = Path(os.environ.get("EVAL_CHECKPOINT_ROOT", "/kaggle/working/checkpoints"))
if not Path("/kaggle/working").exists():
    # Local / non-Kaggle fallback so the script is still runnable for smoke-testing.
    OUTPUT_ROOT = REPO_ROOT / "outputs" / "eval_results_3b"
    CHECKPOINT_ROOT = REPO_ROOT / "outputs" / "checkpoints_3b"


# ── 1. Dependency bootstrap (never touches torch — Kaggle's GPU image ships
#      a CUDA-matched torch build; reinstalling it can silently break CUDA) ──

_PIP_PACKAGES = [
    "transformers>=4.51.0",
    "peft>=0.15.2",
    "datasets>=3.0.0",
    "pydantic>=2.7.0",
    "PyYAML>=6.0.1",
    "huggingface_hub[cli]>=0.24.0",
]


def _module_name(pkg_spec: str) -> str:
    name = pkg_spec.split(">=", 1)[0].split("==", 1)[0].split("[", 1)[0]
    return {"PyYAML": "yaml"}.get(name, name)


def ensure_dependencies() -> None:
    missing = [p for p in _PIP_PACKAGES if importlib.util.find_spec(_module_name(p)) is None]
    if not missing:
        logger.info("All required Python dependencies already present.")
        return
    logger.info("Installing missing dependencies: %s", missing)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *missing],
        check=True,
    )


def ensure_gpu_runtime() -> int:
    """Return the number of visible CUDA devices, failing loudly if none."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script requires the Kaggle 'GPU T4 x2' "
            "accelerator (Settings -> Accelerator -> GPU T4 x2)."
        )
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    logger.info("CUDA devices available: %s", names)
    return count


# ── 2. Optional HF token from Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN) ──


def configure_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        logger.info("Loaded HF_TOKEN from Kaggle Secrets.")
    except Exception:
        logger.info("No HF_TOKEN secret found — proceeding unauthenticated (fine for public repos).")


# ── 3. Checkpoint download (resumable: skips if adapter files already local) ──


def download_checkpoint(repo_id: str, dest: Path) -> Path:
    marker = dest / "adapter_config.json"
    if marker.exists():
        logger.info("[%s] Checkpoint already present at %s", repo_id, dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Downloading to %s ...", repo_id, dest)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(dest),
        token=os.environ.get("HF_TOKEN"),
    )
    return dest


# ── 4. Per-GPU worker: runs scripts/evaluate.py as a subprocess so we reuse
#      the exact same eval path (metrics, per-problem dump, etc.) as everywhere
#      else in the repo — just pinned to one GPU via CUDA_VISIBLE_DEVICES ──


def run_one(config_name: str, checkpoint_dir: Path, gpu_id: int, k: int, force: bool) -> None:
    out_dir = OUTPUT_ROOT / config_name
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        logger.info("[gpu%d][%s] Results already exist — skipping (use --force to redo).", gpu_id, config_name)
        return

    config_path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_dir),
        "--k", str(k),
        "--output_dir", str(out_dir),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info("[gpu%d][%s] Starting evaluation ...", gpu_id, config_name)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    prefix = f"[gpu{gpu_id}][{config_name}]"
    for line in result.stdout.splitlines():
        print(f"{prefix} {line}")
    if result.returncode != 0:
        for line in result.stderr.splitlines()[-40:]:
            print(f"{prefix} STDERR: {line}")
        logger.error("%s FAILED (exit %d)", prefix, result.returncode)
    else:
        logger.info("%s Done.", prefix)


def gpu_worker(gpu_id: int, jobs: list[tuple[str, Path]], k: int, force: bool) -> None:
    for config_name, checkpoint_dir in jobs:
        try:
            run_one(config_name, checkpoint_dir, gpu_id, k, force)
        except Exception:
            logger.exception("[gpu%d][%s] Unexpected error — continuing with remaining models.", gpu_id, config_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 7 3B GRPO checkpoints on Kaggle 2xT4.")
    parser.add_argument("--k", type=int, default=8, help="Sampled completions per problem.")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these config names.")
    parser.add_argument("--force", action="store_true", help="Re-run models that already have saved results.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()

    ensure_dependencies()
    configure_hf_token()
    num_gpus = ensure_gpu_runtime()

    models = MODELS if not args.only else [(c, r) for c, r in MODELS if c in args.only]
    if not models:
        raise SystemExit(f"--only did not match any known config names: {args.only}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %d checkpoint(s) ...", len(models))
    checkpoint_dirs: dict[str, Path] = {}
    for config_name, repo_id in models:
        checkpoint_dirs[config_name] = download_checkpoint(repo_id, CHECKPOINT_ROOT / config_name)

    # Split models round-robin across available GPUs (falls back to 1 GPU if only one is visible).
    num_workers = max(1, min(num_gpus, 2))
    buckets: list[list[tuple[str, Path]]] = [[] for _ in range(num_workers)]
    for i, (config_name, _repo_id) in enumerate(models):
        buckets[i % num_workers].append((config_name, checkpoint_dirs[config_name]))

    logger.info("Distributing %d model(s) across %d GPU(s): %s",
                len(models), num_workers, {i: [c for c, _ in b] for i, b in enumerate(buckets)})

    threads = [
        threading.Thread(target=gpu_worker, args=(gpu_id, jobs, args.k, args.force))
        for gpu_id, jobs in enumerate(buckets)
        if jobs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results_summary: dict[str, Any] = {}
    for config_name, _repo_id in models:
        metrics_path = OUTPUT_ROOT / config_name / "metrics.json"
        if metrics_path.exists():
            results_summary[config_name] = json.loads(metrics_path.read_text())

    summary_path = OUTPUT_ROOT / "all_results_summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    logger.info("Wrote combined summary to %s", summary_path)


if __name__ == "__main__":
    main()
