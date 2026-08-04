"""Evaluate all 7 3B-base-model checkpoints on Modal A100 GPUs in parallel."""
from __future__ import annotations

from scripts.modal_train import app, evaluate_batch

with app.run():
    evaluate_batch.remote(
        config_names=[
            "c1_baseline_3b",
            "c2_hackable_3b",
            "c3_kl_low_3b",
            "c4_kl_med_3b",
            "c5_kl_high_3b",
            "c6_length_cap_3b",
            "c7_kl_cap_combined_3b",
        ],
        k=8,
        seeds=[42],
    )
