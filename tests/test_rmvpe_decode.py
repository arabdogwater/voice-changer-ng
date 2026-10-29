import numpy as np
import torch

from voice_changer.DiffusionSVC.pitchExtractor.rmvpe.rmvpe import RMVPE


class _DecodeOnly(RMVPE):
    def __init__(self):  # only the decoding tables, no network
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))


def test_gpu_decode_matches_numpy_decode():
    r = _DecodeOnly()
    rng = np.random.default_rng(0)
    sal = rng.random((300, 360)).astype(np.float32) ** 8  # peaky, like real salience
    sal[::7] *= 0.01  # some frames below the voicing threshold
    sal[:, 0] += (rng.random(300) > 0.9).astype(np.float32)  # peaks at the lower edge
    sal[:, 359] += (rng.random(300) > 0.9).astype(np.float32)  # and at the upper edge
    want = r.decode(sal.copy(), thred=0.03)
    got = r.decode_gpu(torch.from_numpy(sal), thred=0.03).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-3)
