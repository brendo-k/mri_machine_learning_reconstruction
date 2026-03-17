from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from ml_recon.utils.espirit import espirit_gpu
from ml_recon.utils.image_processing import ifft_2d_img, root_sum_of_squares

SPLITS = ("train", "test", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one reconstructed subject/slice from each BraTS subset using ESPIRiT coil maps."
    )
    parser.add_argument("dataset_root", type=Path, help="Root directory containing generated BraTS datasets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "brats_espirit_examples.png",
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=None,
        help="Slice index to use. Defaults to the center slice of each subject.",
    )
    parser.add_argument(
        "--ifft-rss",
        action="store_true",
        help="Use direct IFFT plus root-sum-of-squares instead of ESPIRiT coil combination.",
    )
    parser.add_argument("--kernel-size", type=int, default=6, help="ESPIRiT kernel size.")
    parser.add_argument("--calib-size", type=int, default=24, help="ESPIRiT calibration region size.")
    parser.add_argument("--sv-thresh", type=float, default=0.02, help="ESPIRiT singular value threshold.")
    parser.add_argument("--crop-thresh", type=float, default=0.95, help="ESPIRiT eigenvalue crop threshold.")
    return parser.parse_args()


def find_variant_roots(dataset_root: Path) -> list[Path]:
    variant_roots = [path for path in sorted(dataset_root.iterdir()) if path.is_dir() and (path / SPLITS[0]).exists()]
    if variant_roots:
        return variant_roots
    return [dataset_root]


def find_subject_file(split_root: Path) -> Path | None:
    subject_files = sorted(split_root.glob("*/*.h5"))
    if subject_files:
        return subject_files[0]
    return None


def decode_contrasts(raw_contrasts: np.ndarray) -> list[str]:
    return [
        contrast.decode("utf-8") if isinstance(contrast, (bytes, np.bytes_)) else str(contrast)
        for contrast in raw_contrasts
    ]


def reconstruct_slice(
    k_space_slice: np.ndarray,
    use_ifft_rss: bool,
    kernel_size: int,
    calib_size: int,
    sv_thresh: float,
    crop_thresh: float,
) -> np.ndarray:
    coil_images = ifft_2d_img(k_space_slice, axes=[-1, -2])
    if use_ifft_rss:
        return root_sum_of_squares(coil_images, coil_dim=0)

    espirit_input = np.transpose(k_space_slice, (1, 2, 0))[None, ...]
    coil_maps = espirit_gpu(
        espirit_input,
        k=kernel_size,
        r=calib_size,
        t=sv_thresh,
        c=crop_thresh,
    )[0, :, :, :, 0]
    if isinstance(coil_maps, torch.Tensor):
        coil_maps = coil_maps.cpu().numpy()
    combined = np.sum(coil_maps.conj() * np.transpose(coil_images, (1, 2, 0)), axis=-1)
    return np.abs(combined)


def load_example(
    subject_file: Path,
    slice_idx: int | None,
    use_ifft_rss: bool,
    kernel_size: int,
    calib_size: int,
    sv_thresh: float,
    crop_thresh: float,
) -> tuple[list[np.ndarray], list[str], int]:
    with h5py.File(subject_file, "r") as handle:
        k_space = handle["k_space"][...]
        contrasts = decode_contrasts(handle["contrasts"][...])

    selected_slice = k_space.shape[0] // 2 if slice_idx is None else slice_idx
    selected_slice = max(0, min(selected_slice, k_space.shape[0] - 1))

    reconstructions = []
    for contrast_idx in range(k_space.shape[1]):
        reconstructions.append(
            reconstruct_slice(
                k_space[selected_slice, contrast_idx],
                use_ifft_rss=use_ifft_rss,
                kernel_size=kernel_size,
                calib_size=calib_size,
                sv_thresh=sv_thresh,
                crop_thresh=crop_thresh,
            )
        )

    return reconstructions, contrasts, selected_slice


def main() -> None:
    args = parse_args()
    variant_roots = find_variant_roots(args.dataset_root)

    examples: list[tuple[str, str, Path, list[np.ndarray], list[str], int]] = []
    for variant_root in variant_roots:
        for split in SPLITS:
            split_root = variant_root / split
            if not split_root.exists():
                continue
            subject_file = find_subject_file(split_root)
            if subject_file is None:
                continue
            reconstructions, contrasts, selected_slice = load_example(
                subject_file,
                slice_idx=args.slice_idx,
                use_ifft_rss=args.ifft_rss,
                kernel_size=args.kernel_size,
                calib_size=args.calib_size,
                sv_thresh=args.sv_thresh,
                crop_thresh=args.crop_thresh,
            )
            examples.append(
                (variant_root.name, split, subject_file, reconstructions, contrasts, selected_slice)
            )

    if not examples:
        raise FileNotFoundError(f"No subject files found under {args.dataset_root}")

    num_rows = len(examples)
    num_cols = len(examples[0][3])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), squeeze=False)

    for row_idx, (variant_name, split, subject_file, reconstructions, contrasts, selected_slice) in enumerate(examples):
        for col_idx, image in enumerate(reconstructions):
            axis = axes[row_idx, col_idx]
            axis.imshow(image, cmap="gray")
            axis.set_xticks([])
            axis.set_yticks([])
            title = contrasts[col_idx] if col_idx < len(contrasts) else f"contrast {col_idx}"
            if row_idx == 0:
                axis.set_title(title)
            if col_idx == 0:
                axis.set_ylabel(
                    f"{variant_name}\n{split}\n{subject_file.parent.name}\nslice {selected_slice}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=60,
                )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
