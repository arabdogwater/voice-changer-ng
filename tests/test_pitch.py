import numpy as np
import torch

from voice_changer.RVC.pitchExtractor.PitchExtractor import F0_MEL_MAX, F0_MEL_MIN, analysis_window, coarse_pitch, pitch_scale, place_tail


def upstream_coarse(f0: np.ndarray) -> np.ndarray:
    # verbatim from upstream RMVPEPitchExtractor / DioPitchExtractor
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(int)


def test_coarse_pitch_matches_upstream():
    f0 = np.concatenate([np.zeros(10), np.linspace(30, 1500, 500)])
    np.testing.assert_array_equal(coarse_pitch(torch.from_numpy(f0)).numpy(), upstream_coarse(f0.copy()))


def test_place_tail_matches_upstream():
    for n_hist, n_new in [(50, 20), (50, 50), (50, 51)]:
        hist = np.arange(n_hist, dtype=np.float64)
        new = 1000 + np.arange(n_new, dtype=np.float64)
        want = hist.copy()
        want[-new.shape[0]:] = new[: want.shape[0]]
        got = place_tail(torch.from_numpy(hist.copy()), torch.from_numpy(new)).numpy()
        np.testing.assert_array_equal(got, want)


def test_pitch_scale_tensor_and_float_agree():
    assert abs(float(pitch_scale(torch.tensor(12.0))) - 2.0) < 1e-6
    assert pitch_scale(-12) == 0.5


def test_analysis_window_keeps_minimum_length():
    # small chunk + long history: start never leaves fewer than min_len samples (RMVPE needs 32 frames)
    assert 6720 - analysis_window(6720, 0.34, 16000, 160, context=1600, min_len=5120) >= 5120
    assert analysis_window(100000, 0.0, 16000, 160, context=1600, min_len=5120) == 0
