"""The TensorRT decoder must match the PyTorch decoder core when replayed inside a CUDA graph, also after
the same engine was used elsewhere in between (another graph, an eager call)."""

import os

import pytest
import torch

pytest.importorskip("tensorrt")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from voice_changer.RVC.inferencer.rvc_models.infer_pack.models import GeneratorNSF  # noqa: E402
from voice_changer.RVC.inferencer.trt_decoder import DecoderCore, TrtEngine, build_engine, export_decoder_onnx  # noqa: E402


def test_trt_decoder_in_cuda_graph(tmp_path):
    torch.manual_seed(0)
    # tiny NSF decoder (same structure as RVC's, small channels/rates) so the engine builds in seconds
    dec = GeneratorNSF(32, "1", [3, 7], [[1, 3, 5], [1, 3, 5]], [4, 4], 64, [8, 8], 16, 16000).eval().cuda()
    for m in dec.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    cl, T = 640, 80
    shapes = ((1, 32, T), (1, 1, T * 16), (1, 16, 1))
    onnx = os.path.join(tmp_path, "dec.onnx")
    export_decoder_onnx(dec, shapes, cl, onnx)
    engine = TrtEngine(build_engine(onnx, fp16=False))
    core = DecoderCore(dec, cl)

    x, har, g = (torch.randn(s, device="cuda") for s in shapes)
    with torch.inference_mode():
        want = core(x, har, g).clone()

        def step(x, har, g):
            y = engine(x=x * 1.0, har=har * 1.0, g=g * 1.0)  # temporaries, like the real pipeline
            junk = torch.randn(1 << 16, device="cuda") * 3  # later kernels that could reuse freed memory
            return y * 1.0, junk

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            step(x, har, g)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, _ = step(x, har, g)
        # someone else uses the same engine with different tensors (another graph key, eager warm-up...)
        other = [torch.randn(s, device="cuda") for s in shapes]
        step(*other)
        del other
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(out, want, atol=2e-3, rtol=0)
