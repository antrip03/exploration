import subprocess
from pathlib import Path

import modal

app = modal.App("grpo-reward-hacking")

model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install([
        "torch==2.7.1",
        "transformers==4.53.2",
        "tensorboard",
        "trl==0.19.1",
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "peft==0.19.1",
        "wandb==0.28.0",
        "bitsandbytes==0.49.2",
        "sentencepiece==0.2.1",
        "huggingface_hub[cli]==0.36.2",
        "ninja==1.13.0",
        "packaging==26.2",
        "pyyaml==6.0.3",
        "pydantic==2.13.4",
        "sentence-transformers==5.6.0",
        "numpy>=1.26.0",
        "safetensors==0.8.0",
    ])
    .run_commands(
        "pip install "
        "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/"
        "wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.7/cu12/abiTRUE/cp311/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    .add_local_dir(".", remote_path="/root/project")
)

CHECKPOINT_REPO_MAP = {
    ("c1_baseline", 42):    "antrip03/grpo-c1_baseline-s42",
    ("c1_baseline", 123):   "antrip03/grpo-c1_baseline-s123",
    ("c1_baseline", 3):     "antrip03/grpo-c1_baseline-s3",
    ("c2_hackable", 42):    "antrip03/grpo-c2_hackable-s42",
    ("c2_hackable", 123):   "antrip03/grpo-c2_hackable-s123",
    ("c2_hackable", 3):     "antrip03/grpo-c2_hackable-s3",
    ("c3_kl_low", 42):      "antrip03/grpo-c3_kl_low",
    ("c3_kl_low", 123):     "antrip03/grpo-c3_kl_low-s123",
    ("c3_kl_low", 3):       "antrip03/grpo-c3_kl_low-s3",
    ("c4_kl_med", 42):      "antrip03/grpo-c4_kl_med",
    ("c4_kl_med", 123):     "antrip03/grpo-c4_kl_med-s123",
    ("c4_kl_med", 3):       "antrip03/grpo-c4_kl_med-s3",
    ("c5_kl_high", 42):     "antrip03/grpo-c5_kl_high",
    ("c5_kl_high", 123):    "antrip03/grpo-c5_kl_high-s123",
    ("c5_kl_high", 3):      "antrip03/grpo-c5_kl_high-s3",
    ("c6_length_cap", 42):  "antrip03/grpo-c6_length_cap",
    ("c6_length_cap", 123): "antrip03/grpo-c6_length_cap-s123",
    ("c6_length_cap", 3):   "antrip03/grpo-c6_length_cap-s3",
    # Seed 456
    ("c1_baseline", 456):        "antrip03/grpo-c1_baseline-s456",
    ("c2_hackable", 456):        "antrip03/grpo-c2_hackable-s456",
    ("c3_kl_low", 456):          "antrip03/grpo-c3_kl_low-s456",
    ("c4_kl_med", 456):          "antrip03/grpo-c4_kl_med-s456",
    ("c5_kl_high", 456):         "antrip03/grpo-c5_kl_high-s456",
    ("c6_length_cap", 456):      "antrip03/grpo-c6_length_cap-s456",
    ("c7_kl_cap_combined", 456): "antrip03/grpo-c7_kl_cap_combined-s456",

    # Length cap sweep (all trained at seed 456)
    ("c6_length_cap_45", 456):   "antrip03/grpo-c6_length_cap_45-s456",
    ("c6_length_cap_100", 456):  "antrip03/grpo-c6_length_cap_100-s456",
    ("c6_length_cap_128", 456):  "antrip03/grpo-c6_length_cap_128-s456",
}


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 6,
    image=image,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def train(config_name: str, overrides: list[str] | None = None):
    import os
    import time
    workdir = Path("/root/project")
    os.environ.setdefault("WANDB_PROJECT", "grpo-reward-hacking")
    cmd = ["python", "scripts/train.py", "--config", f"configs/{config_name}.yaml"]
    if overrides:
        cmd.extend(overrides)
    start = time.time()
    result = subprocess.run(cmd, cwd=workdir)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    print(f"Total wall time: {time.time() - start:.1f}s")


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    image=image,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def evaluate(config_name: str, k: int = 8, seed: int = 42):
    workdir = Path("/root/project")
    repo_id = CHECKPOINT_REPO_MAP[(config_name, seed)]
    checkpoint_dir = workdir / "outputs" / f"{config_name}_s{seed}" / "checkpoint-final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint from {repo_id}...")
    result = subprocess.run([
        "python", "-c",
        f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{repo_id}",
    repo_type="model",
    local_dir="{checkpoint_dir}",
)
print("Download complete.")
"""
    ], cwd=workdir, capture_output=True, text=True)
    print("DOWNLOAD STDOUT:", result.stdout)
    print("DOWNLOAD STDERR:", result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"Download failed for {repo_id}:\n{result.stderr}")

    print(f"Running evaluation for {config_name} seed={seed} k={k}...")
    result = subprocess.run([
        "python", "scripts/evaluate.py",
        "--config", f"configs/{config_name}.yaml",
        "--checkpoint", str(checkpoint_dir),
        "--k", str(k),
        "--output_dir", f"outputs/eval_results/{config_name}_s{seed}",
    ], cwd=workdir, capture_output=True, text=True)
    print("EVAL STDOUT:", result.stdout)
    print("EVAL STDERR:", result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {config_name} seed={seed}:\n{result.stderr[-2000:]}")

    metrics_lines = [
        line.strip() for line in result.stdout.splitlines()
        if ":" in line and not line.startswith("=")
    ]
    return {
        "config_name": config_name,
        "seed": seed,
        "repo_id": repo_id,
        "metrics_output": "\n".join(metrics_lines),
    }


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 2,
    image=image,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def evaluate_batch(config_names: list[str], k: int = 8, seeds: list[int] | None = None):
    if seeds is None:
        seeds = [42, 123]
    jobs = [(cn, sd) for cn in config_names for sd in seeds]
    print(f"Launching {len(jobs)} evaluations in parallel...")
    futures = [evaluate.spawn(config_name=cn, k=k, seed=sd) for cn, sd in jobs]
    outputs = [f.get() for f in futures]
    print("\n" + "=" * 70)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 70)
    for o in outputs:
        print(f"\n--- {o['config_name']} seed={o['seed']} ---")
        print(o['metrics_output'])
    print("=" * 70)
    return outputs


@app.function(
    gpu="A10G",
    timeout=15 * 60,
    image=image,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke_test():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    try:
        import flash_attn
        print("FlashAttention:", flash_attn.__version__)
    except Exception as e:
        raise RuntimeError(f"FlashAttention unavailable: {e}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    print("Model loaded.")
    inputs = tokenizer("What is 2 + 2?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=16)
    print(tokenizer.decode(outputs[0]))


@app.local_entrypoint()
def main(
    config_name: str = "c2_hackable",
    max_steps: int = 1000,
    evaluate_only: bool = False,
    k: int = 8,
    seed: int = 42,
):
    if evaluate_only:
        evaluate.remote(config_name, k=k, seed=seed)
    else:
        overrides = [
            f"training.max_steps={max_steps}",
            f"training.seed={seed}",
            f"training.output_dir=outputs/{config_name}-s{seed}",
        ]
        train.remote(config_name, overrides=overrides)