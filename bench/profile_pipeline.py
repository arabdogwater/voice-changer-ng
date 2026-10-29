"""In-process stage profiler for the RVC pipeline (no HTTP).

    .venv/Scripts/python bench/profile_pipeline.py --slot 0 --chunk 2048 --extra 16384 --f0 rmvpe
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

SERVER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, SERVER)
os.chdir(SERVER)

sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
ap = argparse.ArgumentParser()
ap.add_argument("--slot", type=int, default=0)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--extra", type=int, default=16384)
ap.add_argument("--f0", default="rmvpe")
ap.add_argument("--index-ratio", type=float, default=0.0)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--stages", action="store_true", help="time each stage (forces eager mode: syncs are not capturable)")
ap.add_argument("--no-graphs", action="store_true")
ap.add_argument("--tensorrt", action="store_true", help="use cached TensorRT decoder engines (build them with server/tools/build_trt_engines.py)")
ap.add_argument("--no-streams", action="store_true")
ap.add_argument("--crossfade", type=int, default=1024)
args = ap.parse_args()

from voice_changer.utils.VoiceChangerParams import VoiceChangerParams  # noqa: E402
from voice_changer.ModelSlotManager import ModelSlotManager  # noqa: E402
from voice_changer.RVC.RVCr2 import RVCr2  # noqa: E402
from voice_changer.VoiceChangerV2 import VoiceChangerV2  # noqa: E402

params = VoiceChangerParams(
    model_dir="model_dir", content_vec_500="", content_vec_500_onnx="", content_vec_500_onnx_on=False,
    hubert_base="pretrain/hubert_base.pt", hubert_base_jp="pretrain/rinna_hubert_base_jp.pt", hubert_soft="",
    nsf_hifigan="", sample_mode="", crepe_onnx_full="pretrain/crepe_onnx_full.onnx", crepe_onnx_tiny="pretrain/crepe_onnx_tiny.onnx",
    rmvpe="pretrain/rmvpe.pt", rmvpe_onnx="pretrain/rmvpe.onnx", whisper_tiny="pretrain/whisper_tiny.pt", tensorrt=args.tensorrt,
)
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager  # noqa: E402

VoiceChangerParamsManager.get_instance().setParams(params)
slot = ModelSlotManager.get_instance("model_dir").get_slot_info(args.slot)
model = RVCr2(params, slot)
model.settings.gpu = 0
model.settings.f0Detector = args.f0
model.settings.extraConvertSize = args.extra
model.settings.indexRatio = args.index_ratio
model.settings.silentThreshold = 0
model.initialize()
vc = VoiceChangerV2(params)
vc.setModel(model)
vc.settings.crossFadeOverlapSize = args.crossfade

stats = defaultdict(list)


def timed(name, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        stats[name].append((time.perf_counter() - t) * 1000)
        return r
    return wrap


pl = model.pipeline
pl.use_cuda_graphs = not (args.no_graphs or args.stages)
pl.parallel_streams = pl.parallel_streams and not args.no_streams
if args.stages:
    pl.extractPitch = timed("pitch", pl.extractPitch)
    pl.extractFeatures = timed("embedder", pl.extractFeatures)
    pl.infer = timed("synth", pl.infer)
    if pl.index is not None:
        pl.index.search = timed("index", pl.index.search)
model.inference = timed("model.inference (all)", model.inference)
vc.on_request = timed("on_request (+sola)", vc.on_request)

import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

x, sr = sf.read(os.path.join(SERVER, "test.wav"))
x = x.mean(1) if x.ndim > 1 else x
x = (soxr.resample(x, sr, 48000) * 32767).astype(np.int16)
n_chunks = len(x) // args.chunk
trt = getattr(pl.inferencer.model, "trt_decoder", None)
for i in range(10):  # warm-up; with --tensorrt this also loads the cached engine
    vc.on_request(x[i * args.chunk:(i + 1) * args.chunk])
if trt is not None:
    t0 = time.time()
    while trt.generation == 0 and not trt.failed and time.time() - t0 < 30:
        vc.on_request(x[: args.chunk])
        time.sleep(0.05)
    print("tensorrt engines loaded:", len(trt.engines), "failed:", len(trt.failed))
    for i in range(10):
        vc.on_request(x[i * args.chunk:(i + 1) * args.chunk])
stats.clear()
for i in range(args.n):
    c = x[(i % n_chunks) * args.chunk:(i % n_chunks + 1) * args.chunk]
    vc.on_request(c)

print(f"chunk {args.chunk} ({args.chunk / 48:.1f} ms)  extra {args.extra}  f0 {args.f0}  index {args.index_ratio}  cuda graphs {pl.graphable()}")
for k, v in stats.items():
    v = np.array(v)
    print(f"  {k:24s} p50 {np.percentile(v, 50):7.2f} ms   p95 {np.percentile(v, 95):7.2f} ms")
