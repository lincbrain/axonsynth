"""
Fast B-Spline Rasterizer
========================
Drop-in replacement for synthspline's BSplineCurves.rasterize().

Instead of O(N_curves × N_voxels) distance evaluations, this uses a
segment-based bounding-box sweep:

  1. Tessellate each B-spline into short line segments  (GPU-parallel)
  2. Compute per-segment axis-aligned bounding boxes    (GPU-parallel)
  3. For each batch of B segments, compute (B, P³) distances
     using fully-vectorised GPU ops (no Python inner loop)   (vectorised GPU)
  4. Scatter results into output via scatter_reduce_          (vectorised GPU)

Expected speedup vs synthspline: 2–10× (replaces O(N_seg) Python dispatches
with O(N_seg/batch_size) dispatches)

Usage
-----
    from fast_rasterizer import fast_rasterize
    prob, label, dist = fast_rasterize(curves, shape=(128, 128, 128))

    # Identical interface to:
    prob, label, dist = curves.rasterize(shape, mode='cosine')
"""

from .core import fast_rasterize, tessellate_curves

__all__ = ["fast_rasterize", "tessellate_curves"]
