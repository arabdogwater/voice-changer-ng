"""`resampy.resample`-compatible shim backed by soxr (no numba/llvmlite in the runtime)."""

import numpy as np
import soxr


def resample(x: np.ndarray, sr_orig: int, sr_new: int, filter: str = "kaiser_fast", axis: int = -1) -> np.ndarray:
    if sr_orig == sr_new:
        return x
    quality = "HQ" if filter == "kaiser_fast" else "VHQ"
    y = soxr.resample(np.moveaxis(np.asarray(x, dtype=np.float32), axis, 0), sr_orig, sr_new, quality=quality)
    return np.moveaxis(y, 0, axis)
