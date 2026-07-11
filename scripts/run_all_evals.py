"""Run evaluation on all 12 models in parallel using separate terminal windows."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# All 12 models: (config_name, seed, checkpoint_path)
# Local checkpoints
LOCAL_CHECKPOINTS = {
    ("c1_baseline", 42): "outputs/c1_baseline/checkpoint-final",
    ("c1_baseline", 123): "outputs/c1_baseline-s123/checkpoint-final",
    ("c2_hackable", 42): "outputs/c2_hackable/checkpoint-final",
    ("c2_hackable", 123): "outputs/c2_hackable-s123/checkpoint-final",
    ("c3_kl_low", 42): "outputs/c3_kl_low/checkpoint-final",
    ("c4_kl_med", 42): "outputs/c4_kl_med/checkpoint-final",
    ("c5_kl_high", 42): "outputs/c5_kl_high/checkpoint-final",
    ("c6_length_cap", 42): "outputs/c6_length_cap/checkpoint-final",
}

# Missing checkpoints (seed=123 for c3-c6) — need to download from Hub
MISSING_CHECKPOINTS = [
    ("c3_kl_low", 123, "antrip03/grpo-c3_kl_low-s123"),
    ("c4_kl_med", 123, "antrip03/grpo-c4_kl_med-s123"),
    ("c5_kl_high", 123, "antrip03/grpo-c5_kl_high-s123"),
    ("c6_length_cap", 123, "antrip03/grpo-c6_length_cap-s123"),
]

BASE_DIR = Path(__file__).resolve().parents[1]


def download_missing() -> list[tuple[str, int, str]]:
    """Download missing checkpoints from HuggingFace Hub. Returns list of (config, seed, path)."""
    downloaded = []
    for cfg_name, seed, repo_id in MISSING_CHECKPOINTS:
        dest = BASE_DIR / "outputs" / f"{cfg_name}-s{seed}" / "checkpoint-final"
        if dest.exists() and (dest / "adapter_config.json").exists():
            print(f"[{cfg_name}-s{seed}] Already exists at {dest}, skipping download")
            downloaded.append((cfg_name, seed, str(dest)))
            continue

        print(f"[{cfg_name}-s{seed}] Downloading from {repo_id} to {dest}...")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(dest),
            )
            print(f"[{cfg_name}-s{seed}] Download complete")
            downloaded.append((cfg_name, seed, str(dest)))
        except Exception as e:
            print(f"[{cfg_name}-s{seed}] Download failed: {e}")
    return downloaded


def launch_eval(config_name: str, checkpoint_path: str, k: int = 8) -> subprocess.Popen:
    """Launch an evaluation in a new terminal window."""
    script = BASE_DIR / "scripts" / "evaluate.py"
    config = BASE_DIR / "configs" / f"{config_name}.yaml"
    output_dir = BASE_DIR / "outputs" / f"{config_name}_eval"

    cmd = [
        sys.executable, str(script),
        "--config", str(config),
        "--checkpoint", str(checkpoint_path),
        "--k", str(k),
        "--output_dir", str(output_dir),
    ]

    # Open in a new terminal window
    if sys.platform == "win32":
        # Windows: use start cmd
        return subprocess.Popen(
            ["start", "cmd", "/k"] + [str(c) for c in cmd],
            shell=True,
            cwd=str(BASE_DIR),
        )
    else:
        # Linux/macOS: use gnome-terminal or xterm
        try:
            return subprocess.Popen(
                ["gnome-terminal", "--", "bash", "-c", " ".join(str(c) for c in cmd)],
                cwd=str(BASE_DIR),
            )
        except FileNotFoundError:
            return subprocess.Popen(
                ["xterm", "-e", " ".join(str(c) for c in cmd)],
                cwd=str(BASE_DIR),
            )


def main() -> None:
    k = 8

    # Step 1: Download missing checkpoints
    print("=" * 60)
    print("Step 1: Downloading missing checkpoints from HuggingFace Hub")
    print("=" * 60)
    missing = download_missing()

    # Step 2: Build full list of all 12 models
    all_models = list(LOCAL_CHECKPOINTS.items())  # ((cfg, seed), path)
    for cfg_name, seed, path in missing:
        all_models.append(((cfg_name, seed), path))

    all_models.sort(key=lambda x: (x[0][0], x[0][1]))

    print("\n" + "=" * 60)
    print(f"Step 2: Launching {len(all_models)} evaluations in parallel")
    print("=" * 60)

    processes = []
    for (cfg_name, seed), ckpt_path in all_models:
        if not os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
            print(f"  SKIP {cfg_name:25s} seed={seed:3d} — checkpoint not found at {ckpt_path}")
            continue
        print(f"  LAUNCH {cfg_name:25s} seed={seed:3d}  path={ckpt_path}")
        p = launch_eval(cfg_name, ckpt_path, k=k)
        processes.append((cfg_name, seed, p))

    print("\n" + "=" * 60)
    print(f"Launched {len(processes)} evaluations in separate terminal windows.")
    print("Each will print results when complete. Close terminals when done.")
    print("=" * 60)


if __name__ == "__main__":
    main()