import numpy as np


def _hz_to_mel(f, htk: bool):
    f = np.asanyarray(f, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + f / 700.0)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep, mels)


def _mel_to_hz(m, htk: bool):
    m = np.asanyarray(m, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel(sr: int, n_fft: int, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None, htk: bool = False) -> np.ndarray:
    """Drop-in for `librosa.filters.mel` (norm="slaney"), so the runtime does not need librosa."""
    fmax = sr / 2 if fmax is None else fmax
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin, htk), _hz_to_mel(fmax, htk), n_mels + 2), htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)
