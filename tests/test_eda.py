"""
test_eda.py — Unit tests for src/preprocessing/eda.py

Tests cover:
  - list_images() returns correct counts and structure
  - class_distribution() creates a PNG file
  - show_sample_grid() creates a PNG file
  - check_integrity() correctly identifies broken images
  - pixel_intensity() creates a PNG file
  - image_size_analysis() returns correct stats dict structure
"""
import os
import tempfile

import pytest
from PIL import Image

from src.preprocessing.eda import (
    check_integrity,
    class_distribution,
    image_size_analysis,
    list_images,
    pixel_intensity,
    show_sample_grid,
)


# ---------------------------------------------------------------------------
# Fixture — fake dataset
# ---------------------------------------------------------------------------

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def fake_data(tmp_path):
    """Create a tiny fake dataset with 8 images per class."""
    for cls in CLASS_NAMES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(8):
            img = Image.new("RGB", (150, 120), color=(i * 30, i * 20, i * 10))
            img.save(str(cls_dir / f"img_{i:03d}.jpg"))
    return str(tmp_path)


@pytest.fixture
def fake_image_map(fake_data):
    return list_images(fake_data, class_names=CLASS_NAMES)


# ---------------------------------------------------------------------------
# list_images
# ---------------------------------------------------------------------------

class TestListImages:
    def test_returns_dict_with_all_classes(self, fake_data):
        result = list_images(fake_data, class_names=CLASS_NAMES)
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_correct_image_count(self, fake_data):
        result = list_images(fake_data, class_names=CLASS_NAMES)
        for cls in CLASS_NAMES:
            assert len(result[cls]) == 8

    def test_paths_are_absolute(self, fake_data):
        result = list_images(fake_data, class_names=CLASS_NAMES)
        for paths in result.values():
            for p in paths:
                assert os.path.isabs(p), f"Expected absolute path, got: {p}"

    def test_missing_class_returns_empty_list(self, fake_data):
        result = list_images(fake_data, class_names=["glioma", "missing_class"])
        assert result["missing_class"] == []


# ---------------------------------------------------------------------------
# class_distribution
# ---------------------------------------------------------------------------

class TestClassDistribution:
    def test_creates_png_file(self, fake_image_map, tmp_path):
        save_path = str(tmp_path / "dist.png")
        class_distribution(fake_image_map, save_path=save_path, show=False)
        assert os.path.exists(save_path), "PNG file should be created"
        assert os.path.getsize(save_path) > 0, "PNG file should not be empty"


# ---------------------------------------------------------------------------
# show_sample_grid
# ---------------------------------------------------------------------------

class TestShowSampleGrid:
    def test_creates_png_file(self, fake_image_map, tmp_path):
        save_path = str(tmp_path / "grid.png")
        show_sample_grid(fake_image_map, n_per_class=3, save_path=save_path, show=False)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


# ---------------------------------------------------------------------------
# check_integrity
# ---------------------------------------------------------------------------

class TestCheckIntegrity:
    def test_clean_dataset_returns_empty_dict(self, fake_image_map):
        corrupt = check_integrity(fake_image_map)
        assert corrupt == {} or all(len(v) == 0 for v in corrupt.values())

    def test_detects_zero_byte_file(self, fake_data, tmp_path):
        # Inject a zero-byte file into glioma folder
        bad_path = os.path.join(fake_data, "glioma", "corrupt.jpg")
        with open(bad_path, "wb") as f:
            pass  # zero bytes

        image_map = list_images(fake_data, class_names=CLASS_NAMES)
        corrupt = check_integrity(image_map)
        assert "glioma" in corrupt
        assert bad_path in corrupt["glioma"]

    def test_detects_corrupt_content(self, fake_data):
        # Inject a file with non-image content
        bad_path = os.path.join(fake_data, "notumor", "bad.jpg")
        with open(bad_path, "wb") as f:
            f.write(b"this is not an image at all!!!!")

        image_map = list_images(fake_data, class_names=CLASS_NAMES)
        corrupt = check_integrity(image_map)
        assert "notumor" in corrupt
        assert bad_path in corrupt["notumor"]


# ---------------------------------------------------------------------------
# pixel_intensity
# ---------------------------------------------------------------------------

class TestPixelIntensity:
    def test_creates_png_file(self, fake_image_map, tmp_path):
        save_path = str(tmp_path / "intensity.png")
        pixel_intensity(fake_image_map, n_sample=5, save_path=save_path, show=False)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


# ---------------------------------------------------------------------------
# image_size_analysis
# ---------------------------------------------------------------------------

class TestImageSizeAnalysis:
    def test_returns_dict_for_all_classes(self, fake_image_map, tmp_path):
        save_path = str(tmp_path / "sizes.png")
        stats = image_size_analysis(
            fake_image_map, n_sample=5, save_path=save_path, show=False
        )
        assert set(stats.keys()) == set(CLASS_NAMES)

    def test_stats_have_correct_keys(self, fake_image_map, tmp_path):
        save_path = str(tmp_path / "sizes2.png")
        stats = image_size_analysis(
            fake_image_map, n_sample=5, save_path=save_path, show=False
        )
        expected_keys = {"mean_w", "mean_h", "min_w", "max_w", "min_h", "max_h"}
        for cls, s in stats.items():
            assert expected_keys == set(s.keys()), f"Missing keys for class {cls}"

    def test_dimensions_match_created_images(self, fake_image_map, tmp_path):
        """Images were created at 150×120, stats should reflect this."""
        save_path = str(tmp_path / "sizes3.png")
        stats = image_size_analysis(
            fake_image_map, n_sample=8, save_path=save_path, show=False
        )
        for cls, s in stats.items():
            assert s["mean_w"] == 150, f"{cls}: expected mean_w=150, got {s['mean_w']}"
            assert s["mean_h"] == 120, f"{cls}: expected mean_h=120, got {s['mean_h']}"
