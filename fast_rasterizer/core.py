"""
Fast B-Spline Rasterizer — Core Implementation
===============================================

Algorithm: Segment-Based Fixed-Patch Vectorized Scatter
  1. Tessellate all B-spline curves into short line segments (GPU-parallel)
  2. For each batch of B segments:
       a. Compute a fixed-size patch P×P×P centred on segment midpoint
       b. Evaluate point-to-segment distance for ALL (B, P³) pairs at once
       c. Convert to probability
       d. Scatter (amin/amax/add) into flat output tensors via scatter_reduce_
  3. Reshape flat outputs back to (H, W, D)

Key improvement over per-segment Python loop
---------------------------------------------
Old code: Python  for i in range(N_seg) → O(N_seg) Python↔GPU dispatches
New code: Python  for i in range(N_seg // batch_size) → ~60 dispatches
          GPU     processes batch_size segments simultaneously in each call

Memory per batch (batch=512, r_max=4, P=11):
  pts tensor  : (512, 1331, 3) × 4 B = ~8 MB  (well within 48 GB VRAM)

Complexity: O(N_curves × L/δ × P³ / batch_size)  inner loops, fully vectorised
"""

import math
import torch
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fast_rasterize(
    curves,
    shape:       Tuple[int, int, int],
    mode:        str   = 'cosine',
    seg_length:  float = 1.5,
    batch_size:  int   = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast drop-in replacement for BSplineCurves.rasterize().

    Parameters
    ----------
    curves : BSplineCurves
        synthspline curve collection (torch.nn.ModuleList of BSplineCurve).
    shape : (H, W, D) int tuple
        Spatial dimensions of the output volume.
    mode : 'cosine' | 'binary'
        Probability-falloff mode.
    seg_length : float
        Target segment length in voxels.
    batch_size : int
        Number of segments processed per GPU batch.
        Larger = faster (fewer Python loop iterations) but more VRAM.
        Automatically reduced if a batch would exceed 200 M float elements.

    Returns
    -------
    prob  : (*shape) float32  — probability "at least one axon here"
    label : (*shape) int64    — 1-based index of the closest curve
    dist  : (*shape) float32  — distance to closest centerline (voxels)
    """
    curve_list = list(curves)
    if len(curve_list) == 0:
        dev = torch.device('cpu')
        return (torch.zeros(shape, device=dev),
                torch.zeros(shape, dtype=torch.long, device=dev),
                torch.full(shape, float('inf'), device=dev))

    device = curve_list[0].waypoints.device

    # ── Step 1: tessellate all curves into flat segment tensors ───────────
    all_p0, all_p1, all_r, all_curve_id = tessellate_curves(
        curve_list, seg_length, device
    )
    N_seg = all_p0.shape[0]
    H, W, D = shape

    # ── Step 2: allocate flat output buffers ──────────────────────────────
    N_vox = H * W * D
    # P(no axon) = product of (1 - p_i)  →  store log-sum
    out_log_prob = torch.zeros(N_vox, device=device)
    out_dist     = torch.full((N_vox,), float('inf'), device=device)
    # Encoded: (prob_int * MAX_CURVES + curve_id) → amax gives label of max-prob seg
    MAX_CURVES   = max(len(curve_list) + 1, 10_000)
    out_encoded  = torch.zeros(N_vox, dtype=torch.long, device=device)

    # ── Step 3: batched vectorised scatter ────────────────────────────────
    mode_c = mode[0].lower()

    for seg_start in range(0, N_seg, batch_size):
        seg_end = min(seg_start + batch_size, N_seg)
        B = seg_end - seg_start

        p0_b = all_p0[seg_start:seg_end]   # (B, 3)
        p1_b = all_p1[seg_start:seg_end]   # (B, 3)
        r_b  = all_r [seg_start:seg_end]   # (B,)
        id_b = all_curve_id[seg_start:seg_end]  # (B,) 1-based

        # Patch half-size driven by max radius in this batch
        r_max = r_b.max().item()
        half  = int(math.ceil(r_max)) + 2   # transition zone padding
        P     = 2 * half + 1                 # patch side length

        # Auto-shrink batch if patch tensor would be too large (>200 M elems)
        max_elems = 200_000_000
        if B * P * P * P > max_elems:
            # Recurse with smaller batch
            sub = max(1, max_elems // (P * P * P))
            for ss in range(seg_start, seg_end, sub):
                se = min(ss + sub, seg_end)
                _rasterize_batch_vec(
                    all_p0[ss:se], all_p1[ss:se],
                    all_r[ss:se], all_curve_id[ss:se],
                    shape, mode_c, MAX_CURVES,
                    out_log_prob, out_dist, out_encoded,
                )
        else:
            _rasterize_batch_vec(
                p0_b, p1_b, r_b, id_b,
                shape, mode_c, MAX_CURVES,
                out_log_prob, out_dist, out_encoded,
            )

    # ── Step 4: decode flat outputs ───────────────────────────────────────
    prob  = (1.0 - torch.exp(out_log_prob)).reshape(shape)
    dist  = out_dist.reshape(shape)
    label = (out_encoded % MAX_CURVES).reshape(shape)

    return prob, label, dist


# ─────────────────────────────────────────────────────────────────────────────
# Tessellation
# ─────────────────────────────────────────────────────────────────────────────

def tessellate_curves(curve_list, seg_length: float, device):
    """Sample all B-spline curves into flat segment tensors.

    Returns
    -------
    p0        : (N_seg, 3) float32
    p1        : (N_seg, 3) float32
    radii     : (N_seg,)   float32
    curve_ids : (N_seg,)   int64   — 1-based
    """
    p0_list, p1_list, r_list, id_list = [], [], [], []
    dtype = torch.float32

    for curve_idx, curve in enumerate(curve_list):
        wpts    = curve.waypoints.to(device=device, dtype=dtype)
        arc_len = (wpts[1:] - wpts[:-1]).norm(dim=-1).sum().item()
        n_pts   = max(2, int(math.ceil(arc_len / seg_length)) + 1)
        t_vals  = torch.linspace(0.0, 1.0, n_pts, device=device, dtype=dtype)

        positions = curve.eval_position(t_vals)          # (n_pts, 3)
        r_raw     = curve.eval_radius(t_vals)

        if not torch.is_tensor(r_raw):
            radii = torch.full((n_pts,), float(r_raw), device=device, dtype=dtype)
        else:
            radii = r_raw.to(device=device, dtype=dtype)

        r_mid = 0.5 * (radii[:-1] + radii[1:])          # (n_pts-1,)

        p0_list.append(positions[:-1])
        p1_list.append(positions[1:])
        r_list.append(r_mid)
        id_list.append(torch.full((n_pts - 1,), curve_idx + 1,
                                  device=device, dtype=torch.long))

    return (
        torch.cat(p0_list, dim=0),
        torch.cat(p1_list, dim=0),
        torch.cat(r_list,  dim=0),
        torch.cat(id_list, dim=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised batch rasterizer (no Python inner loop over segments)
# ─────────────────────────────────────────────────────────────────────────────

def _rasterize_batch_vec(
    p0, p1, r, curve_ids,          # (B,3), (B,3), (B,), (B,)
    shape, mode_c, MAX_CURVES,
    out_log_prob, out_dist, out_encoded,
):
    """Process B segments entirely on GPU, no Python loop over segments."""
    B = p0.shape[0]
    H, W, D = shape
    device = p0.device

    r_max = r.max().item()
    half  = int(math.ceil(r_max)) + 2
    P     = 2 * half + 1

    # ── Build offset grid (P, P, P, 3) ──
    ax = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing='ij')
    offsets = torch.stack([gx, gy, gz], dim=-1)   # (P, P, P, 3)
    P3 = P * P * P

    # ── Segment midpoints → integer centre voxel ──
    mid = 0.5 * (p0 + p1)                          # (B, 3)
    ctr = mid.round().long()                        # (B, 3)

    # ── Global coords for every (segment, patch voxel) pair ──
    # coords: (B, P3, 3)
    coords = ctr.unsqueeze(1) + offsets.reshape(1, P3, 3).long()

    # ── In-bounds mask ──
    valid = (
        (coords[:, :, 0] >= 0) & (coords[:, :, 0] < H) &
        (coords[:, :, 1] >= 0) & (coords[:, :, 1] < W) &
        (coords[:, :, 2] >= 0) & (coords[:, :, 2] < D)
    )  # (B, P3) bool

    # ── Vectorised point-to-segment distance: (B, P3) ──
    pts  = coords.float()                                      # (B, P3, 3)
    ab   = p1 - p0                                             # (B, 3)
    ap   = pts - p0.unsqueeze(1)                               # (B, P3, 3)
    ab_sq = (ab * ab).sum(-1).clamp(min=1e-12)                # (B,)
    t    = (ap * ab.unsqueeze(1)).sum(-1) / ab_sq.unsqueeze(1) # (B, P3)
    t    = t.clamp(0.0, 1.0)
    closest = p0.unsqueeze(1) + t.unsqueeze(-1) * ab.unsqueeze(1)  # (B, P3, 3)
    seg_dist = (pts - closest).norm(dim=-1)                    # (B, P3)

    # ── Probability ──
    r_exp = r.unsqueeze(1)                                     # (B, 1)
    if mode_c == 'b':
        seg_prob = (seg_dist <= r_exp).float()
    else:
        seg_prob = _cosine_prob_vec(seg_dist, r_exp)           # (B, P3)

    # ── Zero out invalid voxels ──
    seg_dist = seg_dist.masked_fill(~valid, float('inf'))
    seg_prob = seg_prob.masked_fill(~valid, 0.0)

    # ── Flat linear index (clamp for safety; masked entries harmless) ──
    idx = (
        coords[:, :, 0] * (W * D) +
        coords[:, :, 1] * D +
        coords[:, :, 2]
    ).clamp(0, H * W * D - 1)                                  # (B, P3)

    # Flatten to 1D and filter to valid only ──────────────────────────────
    valid_1d    = valid.reshape(-1)
    idx_1d      = idx.reshape(-1)[valid_1d]
    dist_1d     = seg_dist.reshape(-1)[valid_1d]
    prob_1d     = seg_prob.reshape(-1)[valid_1d]

    # Label encoding: amax over (prob_int * MAX_CURVES + curve_id)
    prob_int    = (prob_1d * 1_000_000).long().clamp(0, 1_000_000)
    cid_1d      = curve_ids.unsqueeze(1).expand(B, P3).reshape(-1)[valid_1d]
    enc_1d      = prob_int * MAX_CURVES + cid_1d

    # ── Scatter reduce ────────────────────────────────────────────────────
    out_dist.scatter_reduce_(0, idx_1d, dist_1d,  reduce='amin', include_self=True)
    out_encoded.scatter_reduce_(0, idx_1d, enc_1d, reduce='amax', include_self=True)
    log1mp = torch.log((1.0 - prob_1d).clamp(min=1e-7))
    out_log_prob.scatter_add_(0, idx_1d, log1mp)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_prob_vec(d: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Cosine falloff matching synthspline's dist_to_prob(mode='cosine').

    d : (...) float  — distances
    r : broadcastable to d — radii
    """
    r     = r.expand_as(d)
    inner = d <= (r - 0.5)
    outer = d >= (r + 0.5)
    trans = ~inner & ~outer

    prob = torch.zeros_like(d)
    prob[inner] = 1.0
    if trans.any():
        prob[trans] = 0.5 * (1.0 + torch.cos(
            math.pi * (d[trans] - r[trans] + 0.5)
        ))
    return prob


# Keep old single-segment helpers for external use / testing
def _point_to_segment_dist(pts, a, b):
    """(N,3) points vs single segment [a,b] → (N,) distances."""
    ab    = b - a
    ap    = pts - a.unsqueeze(0)
    ab_sq = (ab * ab).sum().clamp(min=1e-12)
    t     = (ap * ab.unsqueeze(0)).sum(-1) / ab_sq
    t     = t.clamp(0.0, 1.0)
    closest = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)
    return (pts - closest).norm(dim=-1), t


def _cosine_prob(d: torch.Tensor, r: float) -> torch.Tensor:
    """Scalar-radius version for compatibility."""
    return _cosine_prob_vec(d, torch.tensor(r, device=d.device, dtype=d.dtype))
