"""
Axon Label Generation — Full Density Unidirectional

Generates dense synthetic axon label volumes using B-spline curves
with a fixed z-axis orientation (Bingham distribution) to simulate
coherently aligned axons as seen in light-sheet microscopy.

Parameters used in production:
    shape       : (128, 128, 128)
    voxel_size  : 0.0008 mm  (0.8 μm)  — matched to HiP-CT (0.857 μm)
    tree_density: 10 ** Uniform(6.1, 6.6)  trees/mm³
                  (scaled down from 6.75–7.25 at 0.5 μm to keep the same
                  number of trees per volume: (0.5/0.8)³ ≈ 0.244× factor)
"""
import torch
from synthspline.cli import LabelApp
from synthspline.labelzoo import SynthAxon
from synthspline import random


class FullDensityUnidirectionalAxon(SynthAxon):
    """High-density axon label generator with fixed z-axis orientation.

    All axons are oriented along the z-axis via a Bingham distribution,
    mimicking coherently aligned myelinated axons in light-sheet data.
    """

    class defaults(SynthAxon.defaults):
        orient_variance = random.LogNormal(mean=0.1, std=0.05)
        orient_mixture  = random.RandInt(1, 1)
        tree_density    = 10 ** 6.0   # default; override at construction time

    def forward(self, batch=1):
        """Generate a batch of dense label volumes.

        Returns
        -------
        ReturnedType namedtuple with fields:
            prob, labels, levelmap, nblevelmap, branchmap, skeleton, dist
            Each tensor has shape (B, 1, *spatial).
        """
        if batch > 1:
            out = list(map(lambda x: [x], self()))
            for _ in range(batch - 1):
                for i, x in enumerate(self()):
                    out[i].append(x)
            return self.ReturnedType(*map(torch.cat, out))

        import time

        dim = len(self.shape)

        orient_mix  = self.orient_mixture()
        orient_prob = random.Dirichlet([1] * orient_mix.item()).sample()
        self.orient_mix_sampler = random.Categorical(orient_prob)
        orient_sampler = self.orient_distribution.lower()[0]
        self.orient_sampler = []

        for _ in range(orient_mix):
            if orient_sampler == 'u':
                self.orient_sampler += [random.UniformSphere()]
            elif orient_sampler == 'w':
                mu    = torch.tensor([0., 0., 1.])
                kappa = 1. / self.orient_concentration()
                self.orient_sampler += [random.Watson(mu, kappa)]
            elif orient_sampler == 'b':
                # Fixed rotation: z-axis as primary orientation
                mu    = torch.eye(3)
                kappa = 1. / self.orient_variance(3)
                kappa = kappa.sort(-1, descending=True).values
                kappa[..., -1] = 0
                self.orient_sampler += [random.Bingham(R=mu, Z=kappa)]
            else:
                raise ValueError(f'Unknown orientation sampler: {orient_sampler}')

        print('Orientation: fixed z-axis (Bingham)')

        # Sample number of trees from physical volume × density
        volume = 1
        for s in self.shape:
            volume *= s
        volume *= (self.voxel_size ** dim)
        density  = self.tree_density()
        nb_trees = max(int(volume * density // 1), 1)
        print(f'Sampling {nb_trees} trees  ({density:.2e} trees/mm³)')

        start  = time.time()
        curves, levels, branchings, nb_levels = [], [], [], []
        for _ in range(nb_trees):
            nb_levels1 = self.nb_levels()
            curves1, levels1, branchings1 = self.sample_tree(max_level=nb_levels1)
            nb_levels += [max(levels1)] * len(curves1)
            curves    += curves1
            levels    += levels1
            branchings += branchings1
        print(f'Curves sampled in {time.time() - start:.2f} sec')

        from synthspline.curves import BSplineCurves

        # Use fast vectorised rasterizer (segment-patch scatter_reduce).
        # Falls back to synthspline if fast_rasterizer is unavailable.
        import sys, pathlib
        _exp_dir = str(pathlib.Path(__file__).resolve().parent.parent)
        if _exp_dir not in sys.path:
            sys.path.insert(0, _exp_dir)
        try:
            from fast_rasterizer import fast_rasterize as _fast_rasterize
            _use_fast = True
        except ImportError:
            _use_fast = False

        import synthspline
        synthspline.backend.jitfields = True

        start = time.time()
        curves = BSplineCurves(curves)
        curves.to(self.device)
        if _use_fast:
            prob, labels, dist = _fast_rasterize(curves, self.shape, mode='cosine')
        else:
            prob, labels, dist = curves.rasterize(self.shape, mode='cosine')
        print(f'Curves rasterized in {time.time() - start:.3f} sec')

        start    = time.time()
        levelmap = torch.zeros_like(labels)
        for i, l in enumerate(levels):
            levelmap.masked_fill_(labels == i + 1, l)

        nblevelmap = torch.zeros_like(labels)
        for i, l in enumerate(nb_levels):
            nblevelmap.masked_fill_(labels == i + 1, l)
        print(f'Level maps computed in {time.time() - start:.3f} sec')

        # Skeleton and branchmap are not saved or used downstream (only prob +
        # labels are written to disk and loaded by the DataLoader). Skipping
        # these loops saves ~4000-6000s per volume.
        skeleton  = torch.zeros_like(labels)
        branchmap = torch.zeros_like(prob)

        n_axons = labels.unique().numel() - 1
        print(f'Generated {n_axons} axons')

        return self.ReturnedType(
            prob[None, None],
            labels[None, None],
            levelmap[None, None],
            nblevelmap[None, None],
            branchmap[None, None],
            skeleton[None, None],
            dist[None, None],
        )


if __name__ == '__main__':
    LabelApp(FullDensityUnidirectionalAxon, shape=128).run()
