from torch import device

from const import EmbedderType
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class EmbedderManager:
    currentEmbedder: Embedder | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getEmbedder(cls, embederType: EmbedderType, isHalf: bool, dev: device) -> Embedder:
        cur = cls.currentEmbedder
        # reuse the loaded embedder when switching voices (upstream reloaded it every time)
        if cur is not None and cur.matchCondition(embederType) and cur.dev == dev and cur.isHalf == isHalf:
            return cur
        cls.currentEmbedder = None
        cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        return cls.currentEmbedder

    @classmethod
    def loadEmbedder(cls, embederType: EmbedderType, isHalf: bool, dev: device) -> Embedder:
        from downloader.WeightDownloader import ensureWeights
        from voice_changer.RVC.embedder.HubertLiteEmbedder import HubertLiteEmbedder

        ensureWeights(cls.params, [{"hubert-base-japanese": "hubert_base_jp", "whisper": "whisper_tiny"}.get(embederType, "hubert_base")])

        if embederType in ("hubert_base", "contentvec"):
            # RVC's hubert_base.pt is ContentVec (legacy 500); both share the HuBERT-base architecture
            return HubertLiteEmbedder().loadModel(embederType, cls.params.hubert_base, dev, isHalf)
        elif embederType == "hubert-base-japanese":
            return HubertLiteEmbedder().loadModel(embederType, cls.params.hubert_base_jp, dev, isHalf)
        elif embederType == "whisper":
            from voice_changer.RVC.embedder.Whisper import Whisper

            return Whisper().loadModel(cls.params.whisper_tiny, dev, isHalf)
        else:
            return HubertLiteEmbedder().loadModel("hubert_base", cls.params.hubert_base, dev, isHalf)
