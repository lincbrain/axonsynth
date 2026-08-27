#!/usr/bin/env python3
"""Generate one deterministic dense synthetic axon label volume."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vol-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def split_for_volume(vol_id: int) -> str:
    if not 0 <= vol_id < 600:
        raise ValueError(f"vol-id must be in [0, 600), got {vol_id}")
    if vol_id < 510:
        return "train"
    if vol_id < 570:
        return "val"
    return "test"


def save_nifti_atomic(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(data, affine), temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    from datagen.axon_labels_full_density import FullDensityUnidirectionalAxon

    split = split_for_volume(args.vol_id)
    output_dir = args.output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"vol{args.vol_id:04d}"
    label_path = output_dir / f"{stem}_label.nii.gz"
    probability_path = output_dir / f"{stem}_prob.nii.gz"
    metadata_path = output_dir / f"{stem}_meta.txt"
    if label_path.is_file() and probability_path.is_file() and metadata_path.is_file():
        print(f"{stem} is complete; skipping")
        return 0

    seed = 42_000 + args.vol_id
    np.random.seed(seed % (2**31))
    torch.manual_seed(seed % (2**31))
    density_exponent = float(np.random.uniform(6.1, 6.6))
    tree_density = 10**density_exponent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = FullDensityUnidirectionalAxon(
        shape=(128, 128, 128),
        voxel_size=0.0008,
        tree_density=tree_density,
    ).to(device)
    generator.device = device

    started = time.time()
    with torch.no_grad():
        generated = generator()
    elapsed = time.time() - started
    labels = generated[1][0, 0].cpu().numpy().astype(np.int32)
    probability = generated[0][0, 0].cpu().numpy().astype(np.float32)

    affine = np.diag([0.0008, 0.0008, 0.0008, 1.0])
    save_nifti_atomic(labels, affine, label_path)
    save_nifti_atomic(probability, affine, probability_path)

    axon_instances = int(np.unique(labels).size - 1)
    metadata = (
        f"vol_id: {args.vol_id}\n"
        f"split: {split}\n"
        f"density_exponent: {density_exponent:.4f}\n"
        f"tree_density: {tree_density:.6e}\n"
        f"n_axons: {axon_instances}\n"
        "shape: 128x128x128\n"
        "voxel_size: 0.0008\n"
        f"elapsed_s: {elapsed:.2f}\n"
        f"rng_seed: {seed}\n"
    )
    temporary_metadata = metadata_path.with_suffix(".txt.tmp")
    temporary_metadata.write_text(metadata)
    temporary_metadata.replace(metadata_path)
    print(metadata, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
