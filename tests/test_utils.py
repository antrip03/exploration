"""
tests/test_utils.py
====================
Tests for src/utils.py — utilities for seed, GPU, filesystem, and checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils import (
    CheckpointManager,
    GPUInfo,
    detect_gpu_info,
    ensure_dir,
    get_kaggle_output_dir,
    load_json,
    save_json,
    save_jsonl,
    set_global_seed,
)


class TestSetGlobalSeed:
    def test_runs_without_error(self):
        """set_global_seed completes without raising for any integer seed."""
        set_global_seed(0)
        set_global_seed(42)
        set_global_seed(12345)

    def test_numpy_seed_set(self):
        """NumPy random state is seeded reproducibly."""
        import numpy as np
        set_global_seed(42)
        v1 = np.random.rand()
        set_global_seed(42)
        v2 = np.random.rand()
        assert v1 == v2

    def test_python_random_seed_set(self):
        """Python random state is seeded reproducibly."""
        import random
        set_global_seed(99)
        v1 = random.random()
        set_global_seed(99)
        v2 = random.random()
        assert v1 == v2


class TestDetectGPUInfo:
    def test_returns_gpu_info(self):
        """detect_gpu_info returns a GPUInfo instance."""
        info = detect_gpu_info()
        assert isinstance(info, GPUInfo)

    def test_device_count_non_negative(self):
        """device_count is non-negative."""
        info = detect_gpu_info()
        assert info.device_count >= 0

    def test_memory_non_negative(self):
        """Memory values are non-negative."""
        info = detect_gpu_info()
        assert info.total_memory_gb >= 0.0
        assert info.free_memory_gb >= 0.0


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        """ensure_dir creates a new directory."""
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_directory_ok(self, tmp_path):
        """ensure_dir does not raise if directory already exists."""
        ensure_dir(tmp_path)  # Should not raise
        assert tmp_path.exists()

    def test_returns_path(self, tmp_path):
        """ensure_dir returns a Path object."""
        result = ensure_dir(tmp_path / "test")
        assert isinstance(result, Path)


class TestSaveLoadJson:
    def test_round_trip(self, tmp_path):
        """Data survives a JSON save/load round trip."""
        data = {"condition": "c1", "pass_at_1": 0.75, "tags": ["a", "b"]}
        path = tmp_path / "data.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_file_not_found_raises(self, tmp_path):
        """load_json raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_nested_data_saved(self, tmp_path):
        """Nested dicts and lists are saved correctly."""
        data = {"a": {"b": [1, 2, 3]}}
        path = tmp_path / "nested.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded["a"]["b"] == [1, 2, 3]


class TestSaveJsonl:
    def test_creates_file(self, tmp_path):
        """save_jsonl creates the target file."""
        records = [{"a": 1}, {"a": 2}]
        path = tmp_path / "records.jsonl"
        save_jsonl(records, path)
        assert path.exists()

    def test_correct_line_count(self, tmp_path):
        """save_jsonl writes one line per record."""
        records = [{"x": i} for i in range(5)]
        path = tmp_path / "records.jsonl"
        save_jsonl(records, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_valid_json_lines(self, tmp_path):
        """Each line in the JSONL file is valid JSON."""
        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        path = tmp_path / "records.jsonl"
        save_jsonl(records, path)
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                assert "a" in obj


class TestCheckpointManager:
    def test_no_checkpoints_returns_empty(self, tmp_path):
        """list_checkpoints returns empty list if dir is empty."""
        mgr = CheckpointManager(tmp_path)
        assert mgr.list_checkpoints() == []

    def test_latest_checkpoint_none_when_empty(self, tmp_path):
        """latest_checkpoint returns None if no checkpoints exist."""
        mgr = CheckpointManager(tmp_path)
        assert mgr.latest_checkpoint() is None

    def test_lists_checkpoints_sorted(self, tmp_path):
        """list_checkpoints returns checkpoints sorted by step number."""
        for name in ["checkpoint-100", "checkpoint-500", "checkpoint-200"]:
            (tmp_path / name).mkdir()
        mgr = CheckpointManager(tmp_path)
        checkpoints = mgr.list_checkpoints()
        steps = [CheckpointManager._extract_step(c.name) for c in checkpoints]
        assert steps == sorted(steps)

    def test_latest_checkpoint_is_last(self, tmp_path):
        """latest_checkpoint returns the highest-step checkpoint."""
        for name in ["checkpoint-100", "checkpoint-300", "checkpoint-200"]:
            (tmp_path / name).mkdir()
        mgr = CheckpointManager(tmp_path)
        latest = mgr.latest_checkpoint()
        assert latest is not None
        assert "300" in latest.name

    def test_cleanup_keeps_last_n(self, tmp_path):
        """cleanup_old_checkpoints retains only the last n checkpoints."""
        for name in ["checkpoint-100", "checkpoint-200", "checkpoint-300", "checkpoint-400"]:
            (tmp_path / name).mkdir()
        mgr = CheckpointManager(tmp_path)
        mgr.cleanup_old_checkpoints(keep_last_n=2)
        remaining = mgr.list_checkpoints()
        assert len(remaining) == 2


class TestGetKaggleOutputDir:
    def test_returns_path(self):
        """get_kaggle_output_dir returns a Path object."""
        result = get_kaggle_output_dir()
        assert isinstance(result, Path)
