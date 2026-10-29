"""TensorRT fp16 engine for the NSF-HiFiGAN decoder (the conv-heavy tail of RVC).

The random NSF sine source stays in torch (it is cheap and stochastic); the engine takes the
latent, the harmonic source and the speaker embedding and returns the waveform. Shapes are
static per streaming configuration, engines are cached on disk and built in a background
thread, so the torch path keeps serving audio until the engine is ready.
"""

import hashlib
import os
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mods.log_control import VoiceChangaerLogger
from voice_changer.RVC.inferencer.rvc_models.infer_pack.modules import LRELU_SLOPE

logger = VoiceChangaerLogger.get_instance().getLogger()


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


class DecoderCore(nn.Module):
    """GeneratorNSF.infer_realtime without the source module, with right-aligned adds via F.pad (export friendly)."""

    def __init__(self, dec, convert_length: int):
        super().__init__()
        self.dec = dec
        self.cl = int(convert_length)

    def forward(self, x, har, g):
        dec, cl = self.dec, self.cl
        x = dec.conv_pre(x) + dec.cond(g)
        for i in range(dec.num_upsamples):
            x = x[:, :, -dec.ups_size[i] + (-cl // int(np.prod(dec.upsample_rates[i:]))):]
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_ = dec.ups[i](x)
            xs = dec.noise_convs[i](har[:, :, -cl - int(dec.noise_conv_size[i]):])
            n = max(x_.shape[2], xs.shape[2])
            x = F.pad(x_, (n - x_.shape[2], 0)) + F.pad(xs, (n - xs.shape[2], 0))
            xs = None
            for j in range(dec.num_kernels):
                r = dec.resblocks[i * dec.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / dec.num_kernels
        x = F.leaky_relu(x)
        return torch.tanh(dec.conv_post(x))


class TrtEngine:
    def __init__(self, engine_bytes: bytes):
        import tensorrt as trt

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outputs = {}
        for n in self.names:
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                self.outputs[n] = torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=torch.float32, device="cuda")
                self.context.set_tensor_address(n, self.outputs[n].data_ptr())

    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        keep = []
        for name, t in inputs.items():
            t = t.float().contiguous()
            keep.append(t)
            self.context.set_tensor_address(name, t.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return next(iter(self.outputs.values()))


def build_engine(onnx_path: str, fp16: bool = True) -> bytes:
    import tensorrt as trt

    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = 2  # level 3 (default) builds ~4x slower for ~0.2 ms
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


class TrtDecoderCache:
    """One engine per (model, decoder input shape, convert_length); built off the audio thread."""

    def __init__(self, model_file: str, dec, cache_dir: str):
        self.dec = dec
        self.cache_dir = cache_dir
        with open(model_file, "rb") as f:
            self.model_hash = hashlib.sha1(f.read(1 << 20)).hexdigest()[:12] + f"_{os.path.getsize(model_file)}"
        self.engines: dict[tuple, TrtEngine] = {}
        self.building: set[tuple] = set()
        self.failed: set[tuple] = set()
        self.lock = threading.Lock()
        self.generation = 0  # bumped whenever an engine becomes usable

    def get(self, x: torch.Tensor, har: torch.Tensor, g: torch.Tensor, convert_length: int) -> TrtEngine | None:
        key = (tuple(x.shape), tuple(har.shape), int(convert_length))
        eng = self.engines.get(key)
        if eng is not None or key in self.failed or torch.cuda.is_current_stream_capturing():
            return eng  # builds are only scheduled from eager/warm-up runs, never from inside a capture
        with self.lock:
            if key not in self.building:
                self.building.add(key)
                shapes = (tuple(x.shape), tuple(har.shape), tuple(g.shape))
                threading.Thread(target=self._build, args=(key, shapes, int(convert_length)), daemon=True).start()
        return None

    def _build(self, key, shapes, convert_length):
        tag = f"{self.model_hash}_x{'x'.join(map(str, shapes[0]))}_h{shapes[1][-1]}_c{convert_length}_trt{_trt_version()}"
        plan_path = os.path.join(self.cache_dir, tag + ".engine")
        try:
            torch.cuda.set_stream(torch.cuda.Stream())  # keep this thread off the legacy default stream
            os.makedirs(self.cache_dir, exist_ok=True)
            if not os.path.exists(plan_path):
                logger.info(f"[Voice Changer] building TensorRT decoder engine {tag} (one-off, ~1 min)")
                onnx_path = os.path.join(self.cache_dir, tag + ".onnx")
                export_decoder_onnx(self.dec, shapes, convert_length, onnx_path)
                plan = build_engine(onnx_path)
                tmp = plan_path + ".part"
                with open(tmp, "wb") as f:
                    f.write(plan)
                os.replace(tmp, plan_path)
                os.remove(onnx_path)
            with open(plan_path, "rb") as f:
                engine = TrtEngine(f.read())
            self.engines[key] = engine
            self.generation += 1
            logger.info(f"[Voice Changer] TensorRT decoder engine ready {tag}")
        except Exception as e:
            logger.warning(f"[Voice Changer] TensorRT decoder unavailable for {tag}: {e}")
            self.failed.add(key)


def _trt_version() -> str:
    import tensorrt as trt

    return trt.__version__.split(".")[0] + trt.__version__.split(".")[1]


def export_decoder_onnx(dec, shapes, convert_length: int, path: str):
    import copy

    core = DecoderCore(copy.deepcopy(dec).float().cuda().eval(), convert_length)
    args = tuple(torch.randn(s, device="cuda") for s in shapes)
    with torch.inference_mode(False), torch.no_grad():
        torch.onnx.export(core, args, path, input_names=["x", "har", "g"], output_names=["wav"], opset_version=17, dynamo=False)
