"""Pre-build TensorRT decoder engines while nothing else is using the GPU.

Engines are specific to (voice model, chunk size, crossfade). Building one while audio is running
competes with the voice changer for the GPU and can take several minutes, so do it up front:

    cd server
    uv run --no-sync python tools/build_trt_engines.py --model_dir model_dir --slots 0,3 --chunks 2048,4096

The server (started with --tensorrt true) then finds the cached engine in engine_cache/ and loads
it in about a second.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)
os.chdir(SERVER)

from voice_changer.ModelSlotManager import ModelSlotManager  # noqa: E402
from voice_changer.RVC.RVCr2 import RVCr2  # noqa: E402
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager  # noqa: E402
from voice_changer.VoiceChangerV2 import VoiceChangerV2  # noqa: E402
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="model_dir")
    ap.add_argument("--slots", required=True, help="comma separated slot indexes")
    ap.add_argument("--chunks", default="2048", help="chunk sizes in samples @48 kHz, as set in the UI")
    ap.add_argument("--extra", type=int, default=16384, help="extraConvertSize used in the UI")
    ap.add_argument("--crossfade", type=int, default=4096, help="crossFadeOverlapSize used in the UI")
    ap.add_argument("--f0", default="rmvpe")
    args = ap.parse_args()

    params = VoiceChangerParams(
        model_dir=args.model_dir, content_vec_500="", content_vec_500_onnx="", content_vec_500_onnx_on=False,
        hubert_base="pretrain/hubert_base.pt", hubert_base_jp="pretrain/rinna_hubert_base_jp.pt", hubert_soft="",
        nsf_hifigan="", sample_mode="", crepe_onnx_full="pretrain/crepe_onnx_full.onnx", crepe_onnx_tiny="pretrain/crepe_onnx_tiny.onnx",
        rmvpe="pretrain/rmvpe.pt", rmvpe_onnx="pretrain/rmvpe.onnx", whisper_tiny="pretrain/whisper_tiny.pt", tensorrt=True,
    )
    VoiceChangerParamsManager.get_instance().setParams(params)
    slots = ModelSlotManager.get_instance(args.model_dir)
    rng = np.random.default_rng(0)

    for slot in [int(s) for s in args.slots.split(",")]:
        info = slots.get_slot_info(slot)
        if info is None or info.voiceChangerType != "RVC":
            print(f"slot {slot}: not an RVC model, skipped")
            continue
        model = RVCr2(params, info)
        model.settings.gpu = 0
        model.settings.f0Detector = args.f0
        model.settings.extraConvertSize = args.extra
        model.settings.silentThreshold = 0
        model.initialize()
        trt = getattr(model.pipeline.inferencer.model, "trt_decoder", None)
        if trt is None:
            print("TensorRT is not available (uv sync --extra tensorrt)")
            return
        for chunk in [int(c) for c in args.chunks.split(",")]:
            vc = VoiceChangerV2(params)
            vc.setModel(model)
            vc.settings.crossFadeOverlapSize = args.crossfade
            t0 = time.time()
            before = trt.generation + len(trt.failed)
            for _ in range(3):  # first chunks schedule the build (or load the cached engine)
                vc.on_request((rng.standard_normal(chunk) * 3000).astype(np.int16))
            while trt.generation + len(trt.failed) == before and time.time() - t0 < 1800:
                time.sleep(1)
            state = "ready" if trt.generation + len(trt.failed) > before and not trt.failed else "FAILED"
            print(f"slot {slot} ({info.name}) chunk {chunk}: engine {state} in {time.time() - t0:.0f} s")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
