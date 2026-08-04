"""Kaggle TPU launcher — evaluate the 7 Qwen2.5-3B GRPO checkpoints.

Designed to run inside a Kaggle notebook with the "TPU VM v3-8" accelerator
(torch_xla preinstalled). Loads each LoRA checkpoint onto a single TPU core,
runs the standard greedy + k-sample evaluation, and writes results to
/kaggle/working so they survive as notebook Output after a "Save & Run All"
commit.

Resumable by design: each model's result directory is checked before running,
so re-committing the notebook after an interrupted session only evaluates
whatever is still missing.

Usage (Kaggle notebook cell or terminal, from the repo root):
    python scripts/eval_kaggle_tpu.py
    python scripts/eval_kaggle_tpu.py --k 8 --max-eval-samples 50
    python scripts/eval_kaggle_tpu.py --only c3_kl_low_3b c5_kl_high_3b
    python scripts/eval_kaggle_tpu.py --force   # re-run even if results exist
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("eval_kaggle_tpu")

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


# ── 1. Dependency bootstrap (never touches torch / torch_xla — Kaggle's TPU
#      image ships a matched pair; reinstalling either breaks the runtime) ──

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


_LIBTPU_INDEX = "https://storage.googleapis.com/libtpu-releases/index.html"


def _install_torch_xla() -> None:
    """Install torch_xla pinned to the already-installed torch version.

    Kaggle's TPU image does not always ship torch_xla preinstalled. We pin the
    exact torch version so pip has no reason to touch the already-imported
    torch package mid-process (upgrading torch under an already-running
    interpreter is how you get segfaults, not a working TPU).
    """
    import torch

    torch_version = torch.__version__.split("+")[0]
    logger.info("torch_xla not found; installing torch_xla==%s for TPU support ...", torch_version)
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
            f"torch_xla[tpu]=={torch_version}",
            "-f", _LIBTPU_INDEX,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Automatic torch_xla install failed:\n"
            f"{result.stderr[-2000:]}\n\n"
            "Install it manually in a notebook cell (matching your torch version), then re-run:\n"
            f'  !pip install -q "torch_xla[tpu]=={torch_version}" -f {_LIBTPU_INDEX}\n'
            "If pip reports a version conflict, no matching torch_xla release exists for "
            f"torch=={torch_version} — check https://github.com/pytorch/xla/releases for a "
            "supported pairing and pip install both torch and torch_xla at that pinned version."
        )


def ensure_tpu_runtime() -> Any:
    """Import torch_xla (installing it if needed) and return the XLA device."""
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except ImportError:
        _install_torch_xla()
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "torch_xla was installed but still failed to import. This usually means the "
                "notebook's accelerator isn't actually set to a TPU (Settings -> Accelerator -> "
                "TPU VM v3-8), or a kernel restart is needed — restart the kernel and re-run."
            ) from exc
    device = xm.xla_device()
    logger.info("Using TPU device: %s", device)
    return device


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


# ── 4. TPU-friendly generation: fixed-length padding to avoid per-batch XLA
#      recompilation (the stock src/generation.py pads dynamically per batch,
#      which is fine on CUDA but causes a fresh graph compile — many seconds
#      each — on every distinct prompt-batch shape on TPU) ──


def _tpu_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    device: Any,
    max_new_tokens: int,
    prompt_max_length: int,
    batch_size: int,
    do_sample: bool,
    num_return_sequences: int,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> list[str]:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=prompt_max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        with torch.no_grad():
            output_ids = model.generate(**encoded, **gen_kwargs)
        generated = output_ids[:, prompt_max_length:]
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def _k_completions(model, tokenizer, prompts, *, device, k, **kwargs) -> list[list[str]]:
    flat = _tpu_generate(
        model, tokenizer, prompts,
        device=device, do_sample=True, num_return_sequences=k, **kwargs,
    )
    return [flat[i * k : (i + 1) * k] for i in range(len(prompts))]


# ── 5. Per-model evaluation ──


def evaluate_one(
    config_name: str,
    repo_id: str,
    *,
    device: Any,
    k: int,
    max_eval_samples: int,
    prompt_max_length: int,
    batch_size: int,
    force: bool,
) -> None:
    from src.config import ExperimentConfig
    from src.dataset import load_countdown_dataset, format_prompt
    from src.metrics import compute_all_metrics
    from src.utils import save_json, save_jsonl

    out_dir = OUTPUT_ROOT / config_name
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        logger.info("[%s] Results already exist at %s — skipping (use --force to redo).", config_name, metrics_path)
        return

    config_path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    cfg = ExperimentConfig.from_yaml(config_path)
    if max_eval_samples:
        cfg.dataset.max_eval_samples = max_eval_samples

    checkpoint_dir = download_checkpoint(repo_id, CHECKPOINT_ROOT / config_name)

    logger.info("[%s] Loading base model %s + adapter on TPU ...", config_name, cfg.model.name)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=cfg.model.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    base.to(device)
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()

    _, eval_dataset = load_countdown_dataset(cfg.dataset)
    formatted = eval_dataset.map(lambda row: format_prompt(row, tokenizer), desc=f"[{config_name}] formatting prompts")
    prompts = [str(row["prompt"]) for row in formatted]
    targets = [str(row["target"]) for row in formatted]
    numbers = [list(row["nums"]) for row in formatted]

    gen_common = dict(
        max_new_tokens=cfg.generation.max_new_tokens,
        prompt_max_length=prompt_max_length,
        batch_size=batch_size,
        repetition_penalty=cfg.generation.repetition_penalty,
    )

    logger.info("[%s] Generating greedy completions (%d problems) ...", config_name, len(prompts))
    greedy = _tpu_generate(
        model, tokenizer, prompts, device=device,
        do_sample=False, num_return_sequences=1, **gen_common,
    )

    logger.info("[%s] Generating %d sampled completions per problem ...", config_name, k)
    sampled = _k_completions(
        model, tokenizer, prompts, device=device, k=k,
        temperature=0.7, top_p=cfg.generation.top_p, top_k=cfg.generation.top_k,
        **gen_common,
    )

    metrics = compute_all_metrics(
        sampled, targets, k=k,
        greedy_completions=greedy,
        numbers_per_problem=numbers,
        tokenizer=tokenizer,
    )
    data = metrics.to_dict()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"condition_id": config_name, "checkpoint_repo": repo_id, "metrics": data}, metrics_path)
    per_problem = [
        {"target": t, "nums": n, "prompt": p, "greedy_completion": g, "sampled_completions": s}
        for t, n, p, g, s in zip(targets, numbers, prompts, greedy, sampled)
    ]
    save_jsonl(per_problem, out_dir / "per_problem_results.jsonl")

    print("\n" + "=" * 60)
    print(f"Evaluation Results — {config_name}")
    print("=" * 60)
    for key, value in data.items():
        print(f"  {key:40s} {value:.4f}" if isinstance(value, float) else f"  {key:40s} {value}")
    print("=" * 60)

    # Free TPU HBM before the next model.
    del model, base
    gc.collect()
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        xm.mark_step()
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 7 3B GRPO checkpoints on Kaggle TPU.")
    parser.add_argument("--k", type=int, default=8, help="Sampled completions per problem.")
    parser.add_argument("--max-eval-samples", type=int, default=50, help="Eval problems per model (0 = use config default).")
    parser.add_argument("--prompt-max-length", type=int, default=320, help="Fixed prompt padding length (keeps TPU shapes stable).")
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per generation batch.")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these config names.")
    parser.add_argument("--force", action="store_true", help="Re-run models that already have saved results.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()

    ensure_dependencies()
    configure_hf_token()
    device = ensure_tpu_runtime()

    models = MODELS if not args.only else [(c, r) for c, r in MODELS if c in args.only]
    if not models:
        raise SystemExit(f"--only did not match any known config names: {args.only}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating %d model(s): %s", len(models), [c for c, _ in models])
    results_summary: dict[str, Any] = {}
    for config_name, repo_id in models:
        try:
            evaluate_one(
                config_name, repo_id,
                device=device,
                k=args.k,
                max_eval_samples=args.max_eval_samples,
                prompt_max_length=args.prompt_max_length,
                batch_size=args.batch_size,
                force=args.force,
            )
            metrics_path = OUTPUT_ROOT / config_name / "metrics.json"
            if metrics_path.exists():
                results_summary[config_name] = json.loads(metrics_path.read_text())["metrics"]
        except Exception:
            logger.exception("[%s] Evaluation FAILED — continuing with remaining models.", config_name)

    summary_path = OUTPUT_ROOT / "all_results_summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    logger.info("Wrote combined summary to %s", summary_path)


if __name__ == "__main__":
    main()
