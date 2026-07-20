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
        "accelerate",
        "datasets",
        "peft",
        "wandb",
        "bitsandbytes",
        "sentencepiece",
        "huggingface_hub[cli]",
        "ninja",
        "packaging",
        "pyyaml",
        "pydantic>=2",
        "sentence-transformers>=3.0.0",
        "numpy>=1.26.0",
    ])
    .run_commands(
        "pip install "
        "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/"
        "wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.7/cu12/abiTRUE/cp311/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    .add_local_dir(".", remote_path="/root/project")
)


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 3,
    image=image,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def train(config_name: str, overrides: list[str] | None = None):
    import subprocess
    import torch
    import time
    from pathlib import Path

    print("=" * 80)
    print("MODAL ENVIRONMENT")
    print("=" * 80)

    print(subprocess.check_output(["nvidia-smi"], text=True))

    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("BF16:", torch.cuda.is_bf16_supported())
    print("TF32:", torch.backends.cuda.matmul.allow_tf32)

    print("flash_sdp:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_sdp:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp:", torch.backends.cuda.math_sdp_enabled())

    for pkg in ("flash_attn", "triton", "xformers"):
        try:
            m = __import__(pkg)
            print(f"{pkg}: {m.__version__}")
        except Exception as e:
            print(f"{pkg}: {e}")

    print("=" * 80)
    print("STARTING GPU MONITOR")
    print("=" * 80)

    workdir = Path("/root/project")

    cmd = [
        "python",
        "scripts/train.py",
        "--config",
        f"configs/{config_name}.yaml",
    ]
    if overrides:
        cmd.extend(overrides)

    # Start GPU monitor
    monitor_log = "/tmp/gpu_dmon.log"
    monitor = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "pucm"],
        stdout=open(monitor_log, "w"),
        stderr=subprocess.STDOUT,
    )

    # Launch training
    proc = subprocess.Popen(cmd, cwd=workdir)

    print("Training for 60 seconds...")

    try:
        proc.wait(timeout=60)
        print("Training finished before 60 seconds.")
    except subprocess.TimeoutExpired:
        print("Stopping training after 60 seconds...")
        proc.terminate()

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Stop GPU monitor
    monitor.terminate()
    monitor.wait()

    print("\n" + "=" * 80)
    print("GPU DMON LOG")
    print("=" * 80)

    with open(monitor_log, "r") as f:
        print(f.read())


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
    import os
    workdir = Path("/root/project")

    hf_username = os.environ.get("HF_USERNAME", "antrip03")
    repo_id = f"{hf_username}/grpo-{config_name}-s{seed}"
    checkpoint_dir = workdir / "outputs" / config_name / "checkpoint-final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint from {repo_id}...")
    subprocess.run([
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
    ], cwd=workdir, check=True)

    print(f"Running evaluation for {config_name} with k={k}...")
    subprocess.run([
        "python", "scripts/evaluate.py",
        "--config", f"configs/{config_name}.yaml",
        "--checkpoint", str(checkpoint_dir),
        "--k", str(k),
    ], cwd=workdir, check=True)

    return {"config_name": config_name, "seed": seed, "k": k, "repo_id": repo_id}


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
def evaluate_batch(config_names: list[str], k: int = 8, seeds: list[int] | None = None):
    """Evaluate multiple configs and seeds. Runs each in parallel on separate A100 GPUs."""
    if seeds is None:
        seeds = [42, 123]
    # Build list of (config_name, seed) pairs
    jobs = [(cn, sd) for cn in config_names for sd in seeds]
    print(f"Launching {len(jobs)} evaluations in parallel on A100 GPUs...")
    for cn, sd in jobs:
        print(f"  {cn:25s} seed={sd}")
    # Launch all evaluations in parallel via spawn
    futures = [evaluate.spawn(config_name=cn, k=k, seed=sd) for cn, sd in jobs]
    # Collect results
    outputs = [f.get() for f in futures]
    print("\n" + "=" * 70)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 70)
    for o in outputs:
        print(f"  {o['config_name']:25s} (seed={o['seed']}) -> {o['repo_id']}")
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
    print("Model loaded. Attention:", model.config._attn_implementation)
    inputs = tokenizer("What is 2 + 2?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=16)
    print(tokenizer.decode(outputs[0]))


@app.local_entrypoint()
def main(
    config_name: str = "c2_hackable",
    max_steps: int = 500,
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