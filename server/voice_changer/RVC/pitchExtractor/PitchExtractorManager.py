from typing import Protocol
from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class PitchExtractorManager(Protocol):
    currentPitchExtractor: PitchExtractor | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        cls.currentPitchExtractor = cls.loadPitchExtractor(pitchExtractorType,  gpu)
        return cls.currentPitchExtractor

    @classmethod
    def loadPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        # imported lazily: the CPU extractors pull in pyworld / librosa / onnxruntime, which are optional
        from downloader.WeightDownloader import ensureWeights

        needs = {"crepe_tiny": "crepe_onnx_tiny", "crepe_full": "crepe_onnx_full", "rmvpe": "rmvpe", "rmvpe_onnx": "rmvpe"}.get(pitchExtractorType, "rmvpe")
        ensureWeights(cls.params, [needs])
        if pitchExtractorType == "harvest":
            from voice_changer.RVC.pitchExtractor.HarvestPitchExtractor import HarvestPitchExtractor
            return HarvestPitchExtractor()
        elif pitchExtractorType == "dio":
            from voice_changer.RVC.pitchExtractor.DioPitchExtractor import DioPitchExtractor
            return DioPitchExtractor()
        elif pitchExtractorType == "crepe":
            from voice_changer.RVC.pitchExtractor.CrepePitchExtractor import CrepePitchExtractor
            return CrepePitchExtractor(gpu)
        elif pitchExtractorType == "crepe_tiny":
            from voice_changer.RVC.pitchExtractor.CrepeOnnxPitchExtractor import CrepeOnnxPitchExtractor
            return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_tiny, gpu)
        elif pitchExtractorType == "crepe_full":
            from voice_changer.RVC.pitchExtractor.CrepeOnnxPitchExtractor import CrepeOnnxPitchExtractor
            return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_full, gpu)
        elif pitchExtractorType in ("rmvpe", "rmvpe_onnx"):
            # the torch RMVPE runs in fp16 with GPU decoding and is faster than the onnx export
            from voice_changer.RVC.pitchExtractor.RMVPEPitchExtractor import RMVPEPitchExtractor
            return RMVPEPitchExtractor(cls.params.rmvpe, gpu)
        elif pitchExtractorType == "fcpe":
            from voice_changer.RVC.pitchExtractor.FcpePitchExtractor import FcpePitchExtractor
            return FcpePitchExtractor(gpu)
        else:
            print("[Voice Changer] PitchExctractor not found", pitchExtractorType)
            print("                fallback to rmvpe")
            from voice_changer.RVC.pitchExtractor.RMVPEPitchExtractor import RMVPEPitchExtractor
            return RMVPEPitchExtractor(cls.params.rmvpe, gpu)
