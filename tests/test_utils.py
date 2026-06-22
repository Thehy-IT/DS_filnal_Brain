"""
test_utils.py — Unit tests for src/utils.py

Tests cover:
  - set_seed() → deterministic random outputs
  - compute_class_weights() → correct formula and tensor shape
  - save_history() / load_history() → round-trip JSON
  - get_device() → returns a valid torch.device
"""
import json
import os
import random

import numpy as np
import pytest
import torch

from src.utils import (
    compute_class_weights,
    get_device,
    load_history,
    save_history,
    set_seed,
)


class TestSetSeed:
    def test_numpy_reproducible(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducible(self):
        set_seed(99)
        a = torch.rand(4)
        set_seed(99)
        b = torch.rand(4)
        assert torch.equal(a, b)

    def test_python_random_reproducible(self):
        set_seed(7)
        a = [random.random() for _ in range(5)]
        set_seed(7)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.rand(4)
        set_seed(2)
        b = torch.rand(4)
        assert not torch.equal(a, b)


class TestComputeClassWeights:
    def test_balanced_dataset_equal_weights(self):
        """Balanced classes → all weights should be 1.0."""
        labels = [0, 0, 1, 1, 2, 2, 3, 3]  # 2 per class
        weights = compute_class_weights(labels, num_classes=4)
        assert weights.shape == (4,)
        np.testing.assert_allclose(weights.numpy(), [1.0, 1.0, 1.0, 1.0], atol=1e-5)

    def test_imbalanced_dataset(self):
        """Minority class gets higher weight."""
        labels = [0] * 10 + [1] * 2  # 10 vs 2
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0], "Minority class should have higher weight"

    def test_output_tensor_shape(self):
        labels = list(range(4)) * 5
        weights = compute_class_weights(labels, num_classes=4)
        assert weights.shape == (4,)

    def test_output_is_float_tensor(self):
        labels = [0, 1, 2, 3]
        weights = compute_class_weights(labels, num_classes=4)
        assert weights.dtype == torch.float32

    def test_all_weights_positive(self):
        labels = [0, 0, 1, 2, 2, 2, 3]
        weights = compute_class_weights(labels, num_classes=4)
        assert (weights > 0).all()


class TestSaveLoadHistory:
    def test_round_trip(self, tmp_path):
        history = {
            "train_loss": [0.9, 0.7, 0.5],
            "val_loss": [1.0, 0.8, 0.6],
            "train_acc": [0.6, 0.75, 0.85],
            "val_acc": [0.55, 0.70, 0.80],
        }
        path = str(tmp_path / "history.json")
        save_history(history, path)
        loaded = load_history(path)
        assert loaded == history

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "history.json")
        save_history({"train_loss": [0.5]}, path)
        assert os.path.exists(path)

    def test_json_is_readable(self, tmp_path):
        path = str(tmp_path / "h.json")
        save_history({"val_acc": [0.9]}, path)
        with open(path) as f:
            data = json.load(f)
        assert data == {"val_acc": [0.9]}


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_usable(self):
        device = get_device()
        t = torch.tensor([1.0]).to(device)
        assert t.device.type == device.type
