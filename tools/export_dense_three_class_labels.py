"""Precompute dense 3-class targets from saved dense instance labels.

This utility writes one ``*_gt_class.nii.gz`` file per source ``*_label.nii.gz``
volume into a parallel output directory. To match the live 3-class training
path, it first zeroes labels wherever the paired ``*_prob.nii.gz`` volume is
non-positive, then applies the 6-neighbor shell/interior rule.

The outputs are auxiliary dense targets for inspection or reuse; they do not
replace the instance-label directory that ``AxonSubsetDataset`` currently
consumes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datagen.axon_subset_dataset import build_shell_interior_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Precompute dense 3-class targets from saved dense instance labels.'
    )
    parser.add_argument('--label_dir', required=True, help='Directory with *_label.nii.gz volumes')
    parser.add_argument('--output_dir', required=True, help='Directory for *_gt_class.nii.gz outputs')
    parser.add_argument('--max_volumes', type=int, default=None,
                        help='Optional cap on the number of source volumes to process')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing *_gt_class.nii.gz outputs')
    parser.add_argument('--copy_meta', action='store_true',
                        help='Copy matching *_meta.txt files into the output directory')
    return parser.parse_args()


def save_gt_class(target: np.ndarray, reference_nii: nib.Nifti1Image, out_path: Path) -> None:
    header = reference_nii.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(target, reference_nii.affine, header), str(out_path))


def main() -> None:
    args = parse_args()
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(label_dir.glob('*_label.nii.gz'))
    if not label_paths:
        raise ValueError(f'No *_label.nii.gz files found in {label_dir}')

    if args.max_volumes is not None:
        label_paths = label_paths[:args.max_volumes]

    started = time.perf_counter()
    processed = 0
    written = 0
    skipped = 0
    copied_meta = 0
    total_load_s = 0.0
    total_build_s = 0.0
    total_save_s = 0.0
    total_shell_voxels = 0
    total_interior_voxels = 0

    print(f'Found {len(label_paths)} source label volumes in {label_dir}')
    print(f'Writing dense 3-class targets to {output_dir}')

    for index, label_path in enumerate(label_paths, start=1):
        stem = label_path.name.replace('_label.nii.gz', '')
        prob_path = label_dir / f'{stem}_prob.nii.gz'
        out_path = output_dir / f'{stem}_gt_class.nii.gz'
        meta_src = label_dir / f'{stem}_meta.txt'
        meta_dst = output_dir / meta_src.name

        if not prob_path.exists():
            raise FileNotFoundError(f'Missing paired probability map for {label_path.name}: {prob_path}')

        if out_path.exists() and not args.overwrite:
            skipped += 1
            if args.copy_meta and meta_src.exists() and not meta_dst.exists():
                shutil.copy2(meta_src, meta_dst)
                copied_meta += 1
            print(f'[{index:04d}/{len(label_paths):04d}] skip {out_path.name}')
            continue

        t0 = time.perf_counter()
        label_nii = nib.load(str(label_path))
        prob_nii = nib.load(str(prob_path))
        labels = np.asarray(label_nii.dataobj, dtype=np.int32)
        prob = np.asarray(prob_nii.dataobj, dtype=np.float32)
        t1 = time.perf_counter()
        labels[prob <= 0] = 0
        gt_class = build_shell_interior_target(labels).astype(np.uint8)
        t2 = time.perf_counter()
        save_gt_class(gt_class, label_nii, out_path)
        t3 = time.perf_counter()

        if args.copy_meta and meta_src.exists():
            shutil.copy2(meta_src, meta_dst)
            copied_meta += 1

        shell_voxels = int((gt_class == 1).sum())
        interior_voxels = int((gt_class == 2).sum())
        total_shell_voxels += shell_voxels
        total_interior_voxels += interior_voxels
        total_load_s += (t1 - t0)
        total_build_s += (t2 - t1)
        total_save_s += (t3 - t2)
        processed += 1
        written += 1

        print(
            f'[{index:04d}/{len(label_paths):04d}] wrote {out_path.name} '
            f'load={t1 - t0:.3f}s build={t2 - t1:.3f}s save={t3 - t2:.3f}s '
            f'shell={shell_voxels} interior={interior_voxels}'
        )

    elapsed_s = time.perf_counter() - started
    summary = {
        'label_dir': str(label_dir),
        'output_dir': str(output_dir),
        'n_selected': len(label_paths),
        'n_processed': processed,
        'n_written': written,
        'n_skipped': skipped,
        'copied_meta_files': copied_meta,
        'overwrite': bool(args.overwrite),
        'copy_meta': bool(args.copy_meta),
        'elapsed_s': elapsed_s,
        'mean_load_s': total_load_s / processed if processed else 0.0,
        'mean_build_s': total_build_s / processed if processed else 0.0,
        'mean_save_s': total_save_s / processed if processed else 0.0,
        'mean_total_s': (total_load_s + total_build_s + total_save_s) / processed if processed else 0.0,
        'total_shell_voxels': total_shell_voxels,
        'total_interior_voxels': total_interior_voxels,
        'output_pattern': '*_gt_class.nii.gz',
        'note': 'Auxiliary dense 3-class targets; labels are masked by paired prob<=0 before shell/interior assignment and are not a drop-in replacement for instance-label training inputs.',
    }
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')

    print('')
    print('=== Dense 3-Class Export Summary ===')
    print(json.dumps(summary, indent=2))
    print(f'Summary written to {summary_path}')


if __name__ == '__main__':
    main()
