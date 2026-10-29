"""Silero VAD gate: skip the whole GPU pipeline while nobody is speaking.

Upstream only had an RMS threshold, which either lets keyboard/fan noise through or clips quiet
speech. Silero (MIT, ~2 MB ONNX) runs on the CPU in well under a millisecond per chunk.
"""

import numpy as np
import soxr

WINDOW = 512  # samples @16 kHz per Silero v5 step
CONTEXT = 64


class SileroVAD:
    def __init__(self, model_path: str, threshold: float = 0.4, hangover_ms: float = 400.0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.hangover = int(hangover_ms * 16)  # samples @16 kHz
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.since_speech = 1 << 30
        self.sr = np.array(16000, dtype=np.int64)

    def is_open(self, chunk: np.ndarray, sample_rate: int) -> bool:
        """chunk: float32 mono in [-1, 1] at `sample_rate`. True while speech (or its hangover) is present."""
        x = soxr.resample(chunk.astype(np.float32), sample_rate, 16000) if sample_rate != 16000 else chunk.astype(np.float32)
        self.pending = np.concatenate([self.pending, x])
        speech = False
        while self.pending.shape[0] >= WINDOW:
            w = np.concatenate([self.context, self.pending[None, :WINDOW]], axis=1)
            self.pending = self.pending[WINDOW:]
            prob, self.state = self.session.run(None, {"input": w, "state": self.state, "sr": self.sr})
            self.context = w[:, -CONTEXT:]
            speech |= float(prob[0, 0]) >= self.threshold
        self.since_speech = 0 if speech else self.since_speech + x.shape[0]
        return self.since_speech <= self.hangover
