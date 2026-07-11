"""Evaluate all 12 trained models on Modal A100 GPUs in parallel."""
from __future__ import annotations

from scripts.modal_train import app, evaluate_batch

with app.run():
    evaluate_batch.remote(
        config_names=["c1_baseline", "c2_hackable", "c3_kl_low", "c4_kl_med", "c5_kl_high", "c6_length_cap"],
        k=8,
        seeds=[42, 123],
    )