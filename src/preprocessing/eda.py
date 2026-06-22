"""
eda.py — Exploratory Data Analysis helpers for BrainTumorAI.

Translates and extends the EDA functions from DAKHDL's
`01_EDA_TienXuLy.ipynb` into reusable Python functions:

  list_images()          → List all image files per class
  class_distribution()   → Bar chart of per-class image counts
  show_sample_grid()     → 5×4 sample image grid per class
  check_integrity()      → Detect broken / zero-byte images
  pixel_intensity()      → Histogram of pixel intensity per class
  image_size_analysis()  → Scatter/box of width × height distribution
  full_eda_report()      → Run all checks and save figures to reports/

Usage in notebook:
    from src.preprocessing.eda import full_eda_report
    full_eda_report(data_dir="data/Training")

Note on imports:
  matplotlib is imported lazily (inside each plotting function) so that
  importing this module does NOT trigger the matplotlib font-cache build.
  This avoids PermissionErrors in sandboxed / CI environments.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import data_cfg, train_cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _figures_dir(reports_dir: Optional[str] = None) -> str:
    path = os.path.join(reports_dir or train_cfg.reports_dir, "figures", "eda")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. list_images — per-class image inventory
# ---------------------------------------------------------------------------

def list_images(
    data_dir: str,
    class_names: Optional[List[str]] = None,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Dict[str, List[str]]:
    """
    Build a dict mapping each class name → list of absolute image paths.

    Args:
        data_dir:    Root directory containing class sub-folders.
        class_names: Override default class list from config.
        extensions:  Valid file extensions to include.

    Returns:
        Dict[str, List[str]] — {class_name: [path1, path2, …]}
    """
    class_names = class_names or data_cfg.class_names
    result: Dict[str, List[str]] = {}

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[EDA] WARNING — class folder not found: {cls_dir}")
            result[cls] = []
            continue

        paths = sorted(
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in extensions
        )
        result[cls] = paths
        print(f"[EDA]  {cls:>12} : {len(paths):>5} images")

    total = sum(len(v) for v in result.values())
    print(f"[EDA] Total : {total} images across {len(class_names)} classes")
    return result


# ---------------------------------------------------------------------------
# 2. class_distribution — bar chart
# ---------------------------------------------------------------------------

def class_distribution(
    image_map: Dict[str, List[str]],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot a bar chart of the number of images per class.

    Mirrors DAKHDL's `class_distribution()` visualisation.

    Args:
        image_map:  Output of list_images().
        title:      Figure title.
        save_path:  Path to save the PNG. Auto-generated if None.
        show:       If True, call plt.show() (use in notebooks).
    """
    # Lazy import — avoids font-cache build at module load time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(image_map.keys())
    counts = [len(image_map[c]) for c in classes]
    total = sum(counts)

    # Color palette matching CLASS_INFO in frontend
    palette = ["#EF4444", "#F59E0B", "#10B981", "#6366F1"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(classes, counts, color=palette[: len(classes)], edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Tumour Class")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts) * 1.2)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="600",
        )

    balanced = total / len(classes)
    ax.axhline(y=balanced, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        len(classes) - 0.5, balanced + max(counts) * 0.01,
        f"Balanced: {int(balanced):,}",
        fontsize=8, color="gray",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_figures_dir(), "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[EDA] Class distribution → {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. show_sample_grid — image samples per class
# ---------------------------------------------------------------------------

def show_sample_grid(
    image_map: Dict[str, List[str]],
    n_per_class: int = 5,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Display a grid of n_per_class sample images for each class.

    Mirrors DAKHDL's sample visualisation block (randomly selected).

    Args:
        image_map:    Output of list_images().
        n_per_class:  Number of samples to show per class (default 5).
        save_path:    PNG save path. Auto-generated if None.
        show:         Call plt.show() for notebook display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(image_map.keys())
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 2.5, n_classes * 2.8),
    )
    fig.suptitle("Sample Images per Class", fontsize=13, fontweight="bold", y=1.01)

    palette = ["#EF4444", "#F59E0B", "#10B981", "#6366F1"]

    for row, cls in enumerate(classes):
        paths = image_map[cls]
        rng = np.random.RandomState(42)
        selected = rng.choice(paths, size=min(n_per_class, len(paths)), replace=False)

        for col, path in enumerate(selected):
            ax = axes[row][col] if n_classes > 1 else axes[col]
            try:
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#EEE")
                ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)

            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    cls, fontsize=9, fontweight="bold",
                    color=palette[row % len(palette)],
                    rotation=0, labelpad=50, va="center",
                )

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_figures_dir(), "sample_grid.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[EDA] Sample grid → {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. check_integrity — broken / zero-byte / corrupt images
# ---------------------------------------------------------------------------

def check_integrity(
    image_map: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Detect broken or unreadable images in the dataset.

    Mirrors DAKHDL's integrity check step before training.

    Args:
        image_map: Output of list_images().

    Returns:
        Dict mapping class_name → list of corrupt file paths.
        An empty dict means the dataset is clean.
    """
    corrupt: Dict[str, List[str]] = {}
    total_checked = 0

    for cls, paths in image_map.items():
        bad = []
        for path in paths:
            total_checked += 1
            # Check 1: zero-byte file
            if os.path.getsize(path) == 0:
                bad.append(path)
                continue
            # Check 2: PIL cannot open it
            try:
                with Image.open(path) as img:
                    img.verify()   # Checks for truncation without decoding
            except (UnidentifiedImageError, Exception):
                bad.append(path)

        if bad:
            corrupt[cls] = bad
            print(f"[EDA] ⚠️  {cls}: {len(bad)} corrupt / unreadable image(s)")

    if not corrupt:
        print(f"[EDA] ✅ Integrity OK — all {total_checked} images readable")
    else:
        total_bad = sum(len(v) for v in corrupt.values())
        print(f"[EDA] ❌ Found {total_bad} corrupt image(s) out of {total_checked}")

    return corrupt


# ---------------------------------------------------------------------------
# 5. pixel_intensity — histogram per class
# ---------------------------------------------------------------------------

def pixel_intensity(
    image_map: Dict[str, List[str]],
    n_sample: int = 50,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot per-class pixel intensity histograms (grayscale).

    Samples n_sample images per class for speed.

    Args:
        image_map:  Output of list_images().
        n_sample:   Number of images to sample per class.
        save_path:  PNG save path. Auto-generated if None.
        show:       Call plt.show().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(image_map.keys())
    palette = ["#EF4444", "#F59E0B", "#10B981", "#6366F1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Pixel Intensity Distribution per Class (Grayscale)", fontsize=12, fontweight="bold")

    rng = np.random.RandomState(42)

    for cls, color in zip(classes, palette):
        paths = image_map[cls]
        sample = rng.choice(paths, size=min(n_sample, len(paths)), replace=False)
        all_pixels: List[np.ndarray] = []

        for path in sample:
            try:
                img = np.array(Image.open(path).convert("L"), dtype=np.float32)
                all_pixels.append(img.ravel())
            except Exception:
                pass

        if all_pixels:
            pixels = np.concatenate(all_pixels)
            ax.hist(
                pixels, bins=64, alpha=0.5, color=color,
                label=cls, density=True, histtype="stepfilled",
            )

    ax.set_xlabel("Pixel Intensity (0–255)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_figures_dir(), "pixel_intensity.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[EDA] Pixel intensity → {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. image_size_analysis — width × height distribution
# ---------------------------------------------------------------------------

def image_size_analysis(
    image_map: Dict[str, List[str]],
    n_sample: int = 100,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Dict]:
    """
    Analyse and plot the distribution of image dimensions.

    Mirrors DAKHDL's image size analysis block.

    Returns a dict with per-class statistics:
      {class: {mean_w, mean_h, min_w, max_w, min_h, max_h}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    classes = list(image_map.keys())
    palette = ["#EF4444", "#F59E0B", "#10B981", "#6366F1"]
    stats: Dict[str, Dict] = {}

    rng = np.random.RandomState(42)

    # Collect sizes
    all_widths: Dict[str, List[int]] = {}
    all_heights: Dict[str, List[int]] = {}

    for cls in classes:
        paths = image_map[cls]
        sample = rng.choice(paths, size=min(n_sample, len(paths)), replace=False)
        ws, hs = [], []
        for path in sample:
            try:
                with Image.open(path) as img:
                    ws.append(img.width)
                    hs.append(img.height)
            except Exception:
                pass
        all_widths[cls] = ws
        all_heights[cls] = hs
        stats[cls] = {
            "mean_w": int(np.mean(ws)) if ws else 0,
            "mean_h": int(np.mean(hs)) if hs else 0,
            "min_w": int(np.min(ws)) if ws else 0,
            "max_w": int(np.max(ws)) if ws else 0,
            "min_h": int(np.min(hs)) if hs else 0,
            "max_h": int(np.max(hs)) if hs else 0,
        }
        print(
            f"[EDA]  {cls:>12}: W {stats[cls]['min_w']}–{stats[cls]['max_w']} px "
            f"| H {stats[cls]['min_h']}–{stats[cls]['max_h']} px "
            f"| avg {stats[cls]['mean_w']}×{stats[cls]['mean_h']}"
        )

    # Plot scatter + marginal box plots
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax_scatter = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1])

    fig.suptitle("Image Dimension Analysis", fontsize=12, fontweight="bold")

    # Scatter: width vs height
    for cls, color in zip(classes, palette):
        ax_scatter.scatter(
            all_widths[cls], all_heights[cls],
            c=color, alpha=0.4, s=18, label=cls,
        )
    ax_scatter.axvline(x=224, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax_scatter.axhline(y=224, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax_scatter.set_xlabel("Width (px)")
    ax_scatter.set_ylabel("Height (px)")
    ax_scatter.set_title("Width vs Height")
    ax_scatter.legend(fontsize=8)

    # Box plot: width distribution per class
    box_data = [all_widths[cls] for cls in classes]
    bp = ax_box.boxplot(box_data, labels=classes, patch_artist=True, vert=True)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_box.axhline(y=224, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="Target 224px")
    ax_box.set_xlabel("Class")
    ax_box.set_ylabel("Width (px)")
    ax_box.set_title("Width Distribution per Class")
    ax_box.legend(fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_figures_dir(), "image_sizes.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[EDA] Image size analysis → {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return stats


# ---------------------------------------------------------------------------
# 7. full_eda_report — run all EDA checks at once
# ---------------------------------------------------------------------------

def full_eda_report(
    data_dir: str,
    class_names: Optional[List[str]] = None,
    reports_dir: Optional[str] = None,
    show: bool = False,
) -> Dict:
    """
    Run the complete EDA pipeline and save all figures.

    Steps (from DAKHDL's notebook):
      1. List images per class
      2. Class distribution bar chart
      3. Sample image grid
      4. Integrity check (broken files)
      5. Pixel intensity histograms
      6. Image size scatter / box plots

    Args:
        data_dir:     Path to dataset root (contains class sub-folders).
        class_names:  Override class list.
        reports_dir:  Override reports directory.
        show:         Call plt.show() for each figure (notebook mode).

    Returns:
        Dict with keys: image_map, corrupt_files, size_stats.
    """
    print("\n" + "=" * 60)
    print("  EDA REPORT — BrainTumorAI")
    print("=" * 60)

    # Step 1: Index images
    print("\n[1/6] Indexing images …")
    image_map = list_images(data_dir, class_names=class_names)

    # Step 2: Class distribution
    print("\n[2/6] Class distribution …")
    class_distribution(image_map, show=show)

    # Step 3: Sample grid
    print("\n[3/6] Sample image grid …")
    show_sample_grid(image_map, n_per_class=5, show=show)

    # Step 4: Integrity check
    print("\n[4/6] Integrity check …")
    corrupt = check_integrity(image_map)

    # Step 5: Pixel intensity
    print("\n[5/6] Pixel intensity histograms …")
    pixel_intensity(image_map, show=show)

    # Step 6: Image size analysis
    print("\n[6/6] Image size analysis …")
    size_stats = image_size_analysis(image_map, show=show)

    print("\n" + "=" * 60)
    print(f"  EDA COMPLETE — figures saved to {_figures_dir(reports_dir)}")
    print("=" * 60)

    return {
        "image_map": image_map,
        "corrupt_files": corrupt,
        "size_stats": size_stats,
    }
