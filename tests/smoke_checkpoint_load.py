#!/usr/bin/env python3
"""GPU smoke check for the two accepted release checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datagen.axon_subset_dataset import build_shell_interior_target
from inference.infer_lsm import (
    build_model,
    load_checkpoint,
    model_state_dict,
    postprocess_logits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-checkpoint", type=Path, required=True)
    parser.add_argument("--three-class-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA device")
    device = torch.device("cuda")
    paths = {
        "binary": args.binary_checkpoint,
        "three_class_shell_interior": args.three_class_checkpoint,
    }
    results = {}
    for mode, path in paths.items():
        checkpoint = load_checkpoint(path)
        model = build_model(device, mode)
        model.load_state_dict(model_state_dict(checkpoint), strict=True)
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = model(torch.zeros((1, 1, 32, 32, 32), device=device))
        expected_channels = 1 if mode == "binary" else 3
        if tuple(logits.shape) != (1, expected_channels, 32, 32, 32):
            raise RuntimeError(f"Unexpected {mode} output shape: {tuple(logits.shape)}")
        outputs = postprocess_logits(logits.cpu(), mode, threshold=0.5)
        results[mode] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "output_shape": list(logits.shape),
            "output_keys": sorted(outputs),
        }
        del model, logits, checkpoint
        torch.cuda.empty_cache()

    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1:4, 1:4, 1:4] = 1
    target = build_shell_interior_target(labels)
    if set(np.unique(target)) != {0, 1, 2}:
        raise RuntimeError("Shell/interior target smoke check failed")
    print(json.dumps({"device": torch.cuda.get_device_name(0), "models": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
