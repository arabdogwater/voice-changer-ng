import math

import torch
import torch.nn.functional as F
from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    DeviceChangingException,
    HalfPrecisionChangingException,
    NotEnoughDataExtimateF0,
)
from mods.log_control import VoiceChangaerLogger

from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencerNono import OnnxRVCInferencerNono
from voice_changer.RVC.pipeline.cuda_graph import GraphCache
from voice_changer.RVC.pipeline.gpu_index import GpuIndex

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor, legacy_extract

logger = VoiceChangaerLogger.get_instance().getLogger()


class Pipeline(object):
    """RVC pipeline. Every tensor stays on `device` from input to output; the per-chunk GPU work
    is replayed as a single CUDA graph when possible."""

    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    index: GpuIndex | None

    targetSR: int
    device: torch.device
    isHalf: bool

    def __init__(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        index: GpuIndex | None,
        targetSR,
        device,
        isHalf,
    ):
        self.embedder = embedder
        self.inferencer = inferencer
        self.pitchExtractor = pitchExtractor
        logger.info("GENERATE INFERENCER" + str(self.inferencer))
        logger.info("GENERATE EMBEDDER" + str(self.embedder))
        logger.info("GENERATE PITCH EXTRACTOR" + str(self.pitchExtractor))

        self.index = index

        self.targetSR = targetSR
        self.device = device
        self.isHalf = isHalf

        self.sr = 16000
        self.window = 160
        self.featDtype = torch.float16 if isHalf else torch.float32
        self.use_cuda_graphs = True
        self.graphs = GraphCache()
        self.parallel_streams = device.type == "cuda"
        self._side_stream = torch.cuda.Stream(device) if self.parallel_streams else None

    def getPipelineInfo(self):
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.getEmbedderInfo()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {
            "inferencer": inferencerInfo,
            "embedder": embedderInfo,
            "pitchExtractor": pitchExtractorInfo,
            "isHalf": self.isHalf,
            "index": "gpu-exact" if self.index is not None else "none",
            "cudaGraphs": self.graphable(),
        }

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor
        self.graphs.clear()
        self.graphs.disabled = False

    def extractPitch(self, audio_pad, if_f0, pitchf, f0_up_key, silence_front):
        if if_f0 != 1:
            return None, None
        try:
            if getattr(self.pitchExtractor, "gpu_native", False):
                pitch, pitchf = self.pitchExtractor.extract(audio_pad, pitchf, f0_up_key, self.sr, self.window, silence_front=silence_front)
            else:
                key = float(f0_up_key) if torch.is_tensor(f0_up_key) else f0_up_key
                pitch, pitchf = legacy_extract(self.pitchExtractor, audio_pad, pitchf, key, self.sr, self.window, silence_front)
        except IndexError as e:  # NOQA
            logger.exception(e)
            raise NotEnoughDataExtimateF0()
        return pitch.unsqueeze(0), pitchf.unsqueeze(0).float()

    def extractFeatures(self, feats, embOutputLayer, useFinalProj):
        try:
            return self.embedder.extractFeatures(feats, embOutputLayer, useFinalProj)
        except RuntimeError as e:
            if "HALF" in e.__str__().upper():
                raise HalfPrecisionChangingException()
            elif "same device" in e.__str__():
                raise DeviceChangingException()
            else:
                raise e

    def infer(self, feats, p_len, pitch, pitchf, sid, out_size):
        try:
            return self.inferencer.infer(feats, p_len, pitch, pitchf, sid, out_size)
        except RuntimeError as e:
            if "HALF" in e.__str__().upper():
                print("HalfPresicion Error:", e)
                raise HalfPrecisionChangingException()
            else:
                raise e

    def graphable(self) -> bool:
        return (
            self.use_cuda_graphs
            and not self.graphs.disabled
            and self.device.type == "cuda"
            and getattr(self.pitchExtractor, "gpu_native", False)
            and type(self.inferencer) not in [OnnxRVCInferencer, OnnxRVCInferencerNono]
        )

    @torch.inference_mode()
    def exec(
        self,
        sid,
        audio,  # torch.Tensor [n] @16k on device
        pitchf,  # torch.Tensor [m] on device, or None for models without f0
        feature,  # torch.Tensor [m, C] on device
        f0_up_key,
        index_rate,
        if_f0,
        silence_front,
        embOutputLayer,
        useFinalProj,
        repeat,
        protect=0.5,
        out_size=None,
    ):
        # host-side planning: everything that decides shapes or branches is resolved here
        quality_padding_sec = (repeat * (audio.shape[0] - 1)) / self.sr  # reflect padding must be shorter than the input
        t_pad = round(self.sr * quality_padding_sec)
        t_pad_tgt = round(self.targetSR * quality_padding_sec)
        silence_front = silence_front if repeat == 0 else 0  # RVC Quality on: no history reuse
        out_size = out_size if repeat == 0 else None
        has_f0 = if_f0 == 1 and pitchf is not None
        if has_f0 and repeat != 0:
            pitchf = torch.zeros((audio.shape[0] + 2 * t_pad) // self.window, device=self.device)
        if not has_f0:
            pitchf = torch.zeros(1, device=self.device)
        search_index = self.index is not None and index_rate != 0
        use_protect = protect < 0.5 and search_index and has_f0
        feature = feature.to(self.featDtype)
        # values the user drags around stay tensors, so moving a slider never forces a re-capture
        knobs = torch.tensor([sid, f0_up_key, index_rate, protect], dtype=torch.float32).to(self.device, non_blocking=True)

        trt = getattr(getattr(self.inferencer, "model", None), "trt_decoder", None)
        plan = (audio.shape[0], pitchf.shape[0], tuple(feature.shape), has_f0, silence_front, int(embOutputLayer), bool(useFinalProj), t_pad, t_pad_tgt, out_size, search_index, use_protect, trt.generation if trt is not None else -1)

        def core(audio, pitchf, feature, knobs):
            return self._core(audio, pitchf, feature, knobs, plan)

        if self.graphable():
            try:
                audio1, pitchf_out, feats_out = self.graphs.run(plan, core, [audio, pitchf, feature, knobs])
            except Exception as e:  # anything not capturable: run this pipeline eagerly from now on
                logger.warning(f"[Voice Changer] CUDA graph capture failed, running eagerly: {e}")
                self.graphs.disabled = True
                self.graphs.clear()
                torch.cuda.synchronize()
                audio1, pitchf_out, feats_out = core(audio, pitchf, feature, knobs)
        else:
            audio1, pitchf_out, feats_out = core(audio, pitchf, feature, knobs)
            if torch.isnan(feats_out).all():
                raise DeviceCannotSupportHalfPrecisionException()
        return audio1, (pitchf_out if has_f0 else None), feats_out

    def _core(self, audio, pitchf, feature, knobs, plan):
        """GPU work for one chunk: fixed shapes and no host syncs, so it can be captured as a CUDA graph."""
        _, _, _, has_f0, silence_front, embOutputLayer, useFinalProj, t_pad, t_pad_tgt, out_size, search_index, use_protect, _ = plan
        sid = knobs[0:1].long()
        f0_up_key, index_rate, protect = knobs[1], knobs[2], knobs[3]

        audio_pad = F.pad(audio.unsqueeze(0), (t_pad, t_pad), mode="reflect").squeeze(0) if t_pad > 0 else audio
        p_len = audio_pad.shape[0] // self.window

        # pitch and features are independent: run them on two streams so their small kernels overlap
        if self.parallel_streams and audio_pad.is_cuda:
            main = torch.cuda.current_stream()
            side = self._side_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                pitch, pitchf = self.extractPitch(audio_pad, 1 if has_f0 else 0, pitchf, f0_up_key, silence_front)
            feats = self.extractFeatures(audio_pad.view(1, -1), embOutputLayer, useFinalProj)
            main.wait_stream(side)
        else:
            pitch, pitchf = self.extractPitch(audio_pad, 1 if has_f0 else 0, pitchf, f0_up_key, silence_front)
            feats = self.extractFeatures(audio_pad.view(1, -1), embOutputLayer, useFinalProj)

        if search_index:
            # the history part (silence_front) keeps last chunk's retrieved features
            offset = min(math.floor(silence_front * 16000) // 320, feats.shape[1])  # 320 samples per frame
            retrieved = self.index.search(feats[0, offset:]).to(feats.dtype)
            hist = feature[: offset * 2 : 2].to(feats.dtype) if offset > 0 else feats[0, :0]
            pad = feats.shape[1] - hist.shape[0] - retrieved.shape[0]
            parts = [feats[0, :pad]] if pad > 0 else []
            npy = torch.cat(parts + [hist, retrieved])[-feats.shape[1]:]
            rate = index_rate.to(feats.dtype)
            feats = npy.unsqueeze(0) * rate + (1 - rate) * feats
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if use_protect:
            feats0 = feats.clone()

        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None:
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
        feats_len = feats.shape[1]
        if pitch is not None:
            pitch, pitchf = pitch[:, -feats_len:], pitchf[:, -feats_len:]

        # where f0 is unvoiced, mix back the un-retrieved features (upstream behaviour)
        if use_protect:
            pitchff = torch.where(pitchf > 0, torch.ones_like(pitchf), protect.expand_as(pitchf)).unsqueeze(-1)
            feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

        if type(self.inferencer) in [OnnxRVCInferencer, OnnxRVCInferencerNono]:
            npyOffset = math.floor(silence_front * 16000) // 360
            feats = feats[:, npyOffset * 2 :, :]  # NOQA
            feats_len = feats.shape[1]
            if pitch is not None:
                pitch, pitchf = pitch[:, -feats_len:], pitchf[:, -feats_len:]
        p_len = torch.full((1,), feats_len, device=self.device, dtype=torch.long)

        audio1 = self.infer(feats, p_len, pitch, pitchf, sid, out_size)
        if t_pad_tgt != 0:
            audio1 = audio1[t_pad_tgt:-t_pad_tgt]
        pitchf_out = pitchf.squeeze(0) if pitchf is not None else torch.zeros(1, device=self.device)
        return audio1, pitchf_out, feats.squeeze(0)

    def __del__(self):
        del self.embedder
        del self.inferencer
        del self.pitchExtractor
        print('Pipeline has been deleted')
