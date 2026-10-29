"""
VoiceChangerV2向け
"""

from dataclasses import asdict
import numpy as np
import torch
from data.ModelSlot import RVCModelSlot
from mods.log_control import VoiceChangaerLogger

from voice_changer.RVC.RVCSettings import RVCSettings
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.utils.VoiceChangerModel import (
    AudioInOut,
    PitchfInOut,
    FeatureInOut,
    VoiceChangerModel,
)
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.RVC.onnxExporter.export2onnx import export2onnx
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.RVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline

from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    PipelineCreateException,
    PipelineNotInitializedException,
)
import torchaudio

logger = VoiceChangaerLogger.get_instance().getLogger()


class RVCr2(VoiceChangerModel):
    def __init__(self, params: VoiceChangerParams, slotInfo: RVCModelSlot):
        logger.info("[Voice Changer] [RVCr2] Creating instance ")
        self.voiceChangerType = "RVC"

        self.deviceManager = DeviceManager.get_instance()
        EmbedderManager.initialize(params)
        PitchExtractorManager.initialize(params)
        self.settings = RVCSettings()
        self.settings.vad = int(getattr(params, "vad", False))
        self.params = params
        # self.pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector, self.settings.gpu)

        self.pipeline: Pipeline | None = None

        self.audio_buffer: AudioInOut | None = None
        self.pitchf_buffer: PitchfInOut | None = None
        self.feature_buffer: FeatureInOut | None = None
        self.prevVol = 0.0
        self.slotInfo = slotInfo
        # self.initialize()

    def initialize(self):
        logger.info("[Voice Changer][RVCr2] Initializing... ")

        # pipelineの生成
        try:
            self.pipeline = createPipeline(self.params, self.slotInfo, self.settings.gpu, self.settings.f0Detector)
        except PipelineCreateException as e:  # NOQA
            logger.error("[Voice Changer] pipeline create failed. check your model is valid.")
            return

        # その他の設定
        self.settings.tran = self.slotInfo.defaultTune
        self.settings.indexRatio = self.slotInfo.defaultIndexRatio
        self.settings.protect = self.slotInfo.defaultProtect
        logger.info("[Voice Changer] [RVC] Initializing... done")

    def setSamplingRate(self, inputSampleRate, outputSampleRate):
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate
        # self.initialize()

    def update_settings(self, key: str, val: int | float | str):
        logger.info(f"[Voice Changer][RVC]: update_settings {key}:{val}")
        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "gpu":
                self.deviceManager.setForceTensor(False)
                self.initialize()
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
            if key == "f0Detector" and self.pipeline is not None:
                pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector, self.settings.gpu)
                self.pipeline.setPitchExtractor(pitchExtractor)
        else:
            return False
        return True

    def get_info(self):
        data = asdict(self.settings)
        if self.pipeline is not None:
            pipelineInfo = self.pipeline.getPipelineInfo()
            data["pipelineInfo"] = pipelineInfo
        else:
            data["pipelineInfo"] = "None"
        return data

    def get_processing_sampling_rate(self):
        return self.slotInfo.samplingRate

    def _resampler(self, orig: int, new: int, device: torch.device):
        key = (orig, new, str(device))
        cache = self.__dict__.setdefault("_resamplers", {})
        if key not in cache:
            cache[key] = torchaudio.transforms.Resample(orig, new).to(device)
        return cache[key]

    def _resample(self, x: torch.Tensor, orig: int, new: int) -> torch.Tensor:
        if orig == new:
            return x
        return self._resampler(orig, new, x.device)(x.unsqueeze(0)).squeeze(0)

    def _vad_open(self, chunk: np.ndarray) -> bool:
        if not self.settings.vad:
            return True
        vad = self.__dict__.get("_vad")
        if vad is None:
            from downloader.WeightDownloader import ensureWeights, weightPath
            from voice_changer.utils.vad import SileroVAD

            try:
                ensureWeights(self.params, ["silero_vad"])
                vad = self._vad = SileroVAD(weightPath(self.params, "silero_vad"))
            except Exception as e:  # never let the gate take the voice changer down
                logger.warning(f"[Voice Changer] VAD unavailable, gate disabled: {e}")
                self.settings.vad = 0
                return True
        vad.threshold = self.settings.vadThreshold
        return vad.is_open(chunk, self.inputSampleRate)

    def generate_input(
        self,
        newData: torch.Tensor,
        crossfadeSize: int,
        solaSearchFrame: int,
        extra_frame: int,
    ):
        """Append the new 16 kHz chunk to the GPU ring buffers and return the window to convert."""
        dev = newData.device
        inputSize = newData.shape[0]
        newFeatureLength = inputSize // 160  # hopsize:=160
        emb = self.slotInfo.embChannels

        if self.audio_buffer is not None and self.audio_buffer.device == dev:
            self.audio_buffer = torch.cat([self.audio_buffer, newData])
            if self.slotInfo.f0:
                self.pitchf_buffer = torch.cat([self.pitchf_buffer, torch.zeros(newFeatureLength, device=dev)])
            self.feature_buffer = torch.cat([self.feature_buffer, torch.zeros(newFeatureLength, emb, device=dev, dtype=self.feature_buffer.dtype)])
        else:
            self.audio_buffer = newData
            self.pitchf_buffer = torch.zeros(newFeatureLength, device=dev) if self.slotInfo.f0 else None
            self.feature_buffer = torch.zeros(newFeatureLength, emb, device=dev)

        convertSize = inputSize + crossfadeSize + solaSearchFrame + extra_frame
        if convertSize % 160 != 0:  # the model output is cut at the hop size, so round up
            convertSize = convertSize + (160 - (convertSize % 160))
        outSize = int(((convertSize - extra_frame) / 16000) * self.slotInfo.samplingRate)

        # not enough history yet: pad with zeros
        if self.audio_buffer.shape[0] < convertSize:
            self.audio_buffer = torch.cat([torch.zeros(convertSize, device=dev), self.audio_buffer])
            if self.slotInfo.f0:
                self.pitchf_buffer = torch.cat([torch.zeros(convertSize // 160, device=dev), self.pitchf_buffer])
            self.feature_buffer = torch.cat([torch.zeros(convertSize // 160, emb, device=dev, dtype=self.feature_buffer.dtype), self.feature_buffer])

        featureOffset = -convertSize // 160
        self.audio_buffer = self.audio_buffer[-convertSize:]
        if self.slotInfo.f0:
            self.pitchf_buffer = self.pitchf_buffer[featureOffset:]
        self.feature_buffer = self.feature_buffer[featureOffset:]

        return (
            self.audio_buffer,
            self.pitchf_buffer,
            self.feature_buffer,
            convertSize,
            outSize,
        )

    def inference(self, receivedData: AudioInOut, crossfade_frame: int, sola_search_frame: int):
        if self.pipeline is None:
            logger.info("[Voice Changer] Pipeline is not initialized.")
            raise PipelineNotInitializedException()

        device = self.pipeline.device
        chunk = receivedData.astype(np.float32) / 32768.0
        # silence gate on the host copy: no GPU sync needed to decide
        vol = float(np.sqrt(np.square(chunk).mean())) if chunk.size else 0.0

        # everything below runs at 16 kHz on the GPU (pitch, embedder, index), the model decodes at its own rate
        audio16 = self._resample(torch.from_numpy(chunk).to(device, non_blocking=True), self.inputSampleRate, 16000)
        crossfade_frame = int((crossfade_frame / self.inputSampleRate) * 16000)
        sola_search_frame = int((sola_search_frame / self.inputSampleRate) * 16000)
        extra_frame = int((self.settings.extraConvertSize / self.inputSampleRate) * 16000)

        audio, pitchf, feature, convertSize, outSize = self.generate_input(audio16, crossfade_frame, sola_search_frame, extra_frame)

        if vol < self.settings.silentThreshold or not self._vad_open(chunk):
            return np.zeros(convertSize).astype(np.int16) * np.sqrt(vol)

        repeat = 1 if self.settings.rvcQuality else 0
        if_f0 = 1 if self.slotInfo.f0 else 0

        try:
            audio_out, self.pitchf_buffer, self.feature_buffer = self.pipeline.exec(
                self.settings.dstId,
                audio,
                pitchf,
                feature,
                self.settings.tran,
                self.settings.indexRatio,
                if_f0,
                self.settings.extraConvertSize / self.inputSampleRate if self.settings.silenceFront else 0.0,  # history length in seconds
                self.slotInfo.embOutputLayer,
                self.slotInfo.useFinalProj,
                repeat,
                self.settings.protect,
                outSize,
            )
            # keep upstream's loudness behaviour: int16 scale times sqrt(input rms)
            out = audio_out[-outSize:].float() * (32767.5 * np.sqrt(vol))
            out = self._resample(out, self.slotInfo.samplingRate, self.outputSampleRate)
            return out.cpu().numpy()
        except DeviceCannotSupportHalfPrecisionException as e:  # NOQA
            logger.warn("[Device Manager] Device cannot support half precision. Fallback to float....")
            self.deviceManager.setForceTensor(True)
            self.initialize()

        return

    def __del__(self):
        del self.pipeline

        # print("---------- REMOVING ---------------")

        # remove_path = os.path.join("RVC")
        # sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        # for key in list(sys.modules):
        #     val = sys.modules.get(key)
        #     try:
        #         file_path = val.__file__
        #         if file_path.find("RVC" + os.path.sep) >= 0:
        #             # print("remove", key, file_path)
        #             sys.modules.pop(key)
        #     except Exception:  # type:ignore
        #         # print(e)
        #         pass

    def export2onnx(self):
        modelSlot = self.slotInfo

        if modelSlot.isONNX:
            logger.warn("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": ""}

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        torch.cuda.empty_cache()
        self.initialize()

        output_file_simple = export2onnx(self.settings.gpu, modelSlot)

        return {
            "status": "ok",
            "path": f"/tmp/{output_file_simple}",
            "filename": output_file_simple,
        }

    def get_model_current(self):
        return [
            {
                "key": "defaultTune",
                "val": self.settings.tran,
            },
            {
                "key": "defaultIndexRatio",
                "val": self.settings.indexRatio,
            },
            {
                "key": "defaultProtect",
                "val": self.settings.protect,
            },
        ]
