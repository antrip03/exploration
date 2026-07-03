import subprocess
from pathlib import Path

import modal

app = modal.App("grpo-reward-hacking")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "build-essential",
    )
    .pip_install(
        [
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
            "huggingface_hub",
            "ninja",
            "packaging",
            "pyyaml",
            "pydantic>=2",
        ]
    )
    .run_commands(
        "pip install "
        "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/"
        "wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.7/cu12/abiTRUE/cp311/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
    )
)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def train(
    config_name: str,
    overrides: list[str] | None = None,
):

    workdir = Path("/root/project")

    cmd = [
        "python",
        "scripts/train.py",
        "--config",
        f"configs/{config_name}.yaml",
    ]

    if overrides:
        cmd.extend(overrides)

    subprocess.run(
        cmd,
        cwd=workdir,
        check=True,
    )


@app.function(
    gpu="A10G",
    timeout=15 * 60,
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
)
def smoke_test():

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    try:
        import flash_attn
        print("✓ FlashAttention:", flash_attn.__version__)
    except Exception as e:
        raise RuntimeError(f"FlashAttention unavailable: {e}")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )

    print("✓ Model loaded")
    print("Attention implementation:", model.config._attn_implementation)

    inputs = tokenizer(
        "What is 2 + 2?",
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
    )

    print(tokenizer.decode(outputs[0]))


@app.local_entrypoint()
def main(
    config_name: str = "c2_hackable",
    max_steps: int = 50,
):
    train.remote(
        config_name,
        overrides=[f"training.max_steps={max_steps}"],
    )