import numpy as np
import torch

from voice_changer.RVC.pipeline import gpu_index
from voice_changer.RVC.pipeline.gpu_index import GpuIndex


def test_exact_nearest_neighbour(monkeypatch):
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((5000, 64)).astype(np.float32)
    q = rng.standard_normal((37, 64)).astype(np.float32)
    # force several score chunks so the running-argmax merge is exercised
    monkeypatch.setattr(gpu_index, "_MAX_SCORE_ELEMS", 37 * 700)
    idx = GpuIndex(vecs, torch.device("cpu"), half=False)
    got = idx.search(torch.from_numpy(q)).numpy()
    want = vecs[((q[:, None, :] - vecs[None]) ** 2).sum(-1).argmin(1)]
    np.testing.assert_array_equal(got, want)
