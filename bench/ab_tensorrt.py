"""Interleaved A/B of the TensorRT decoder vs the PyTorch decoder in one process.

The GPU is usually shared with the desktop, so back-to-back runs drift; alternating blocks of
chunks exposes both variants to the same background load.

    .venv/Scripts/python bench/ab_tensorrt.py --chunk 2048 --extra 16384
"""

import os
import runpy
import sys
import time

import numpy as np
import torch

args = sys.argv[1:]
block = 50
rounds = 8
sys.argv = [sys.argv[0], "--n", "1", "--tensorrt"] + args
g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_pipeline.py"), run_name="__main__")
vc, x, pl = g["vc"], g["x"], g["pl"]
chunk = g["args"].chunk
model = pl.inferencer.model
trt = model.trt_decoder
if not trt.engines:
    sys.exit("no TensorRT engine for this config; build it with server/tools/build_trt_engines.py")

lat = {"pytorch": [], "tensorrt": []}
n_chunks = len(x) // chunk
i = 0
for r in range(rounds):
    for name in ("pytorch", "tensorrt"):
        model.trt_decoder = trt if name == "tensorrt" else None
        for k in range(block):
            c = x[(i % n_chunks) * chunk:(i % n_chunks + 1) * chunk]
            i += 1
            torch.cuda.synchronize()
            t = time.perf_counter()
            vc.on_request(c)
            if k >= 5:  # skip the first chunks after a switch (graph re-selection)
                lat[name].append((time.perf_counter() - t) * 1000)
for name, v in lat.items():
    print(f"{name:9s} p50 {np.percentile(v, 50):6.2f} ms   p95 {np.percentile(v, 95):6.2f} ms   (n={len(v)})")

# quality: stream the whole file through each decoder and keep the audio for bench/compare_audio.py
import soundfile as sf  # noqa: E402

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
extra = g["args"].extra
for name in ("pytorch", "tensorrt"):
    model.trt_decoder = trt if name == "tensorrt" else None
    for k in range(10):  # settle buffers/graphs after the switch
        vc.on_request(x[k * chunk:(k + 1) * chunk])
    pieces = [vc.on_request(x[k * chunk:(k + 1) * chunk])[0] for k in range(n_chunks)]
    sf.write(os.path.join(out_dir, f"ab_{name}_c{chunk}_e{extra}.wav"), np.concatenate(pieces).astype(np.int16), 48000)
print(f"wrote out/ab_pytorch_c{chunk}_e{extra}.wav and out/ab_tensorrt_c{chunk}_e{extra}.wav")
