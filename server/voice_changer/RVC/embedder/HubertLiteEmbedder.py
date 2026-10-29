import torch
from torch import device

from const import EmbedderType
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.hubert_lite import load_hubert_lite


class HubertLiteEmbedder(Embedder):
    """HuBERT / ContentVec without fairseq (fused attention, fp16, optional CUDA graphs)."""

    def loadModel(self, embedderType: EmbedderType, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().setProps(embedderType, file, dev, isHalf)
        self.model = load_hubert_lite(file, dev, isHalf)
        return self

    def extractFeatures(self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True) -> torch.Tensor:
        # v1 models: layer 9 + final_proj (256), v2 models: layer 12 (768)
        dtype = next(self.model.parameters()).dtype
        return self.model(feats.to(self.dev, dtype), int(embOutputLayer), bool(useFinalProj))

    def setHalf(self, isHalf: bool):
        self.isHalf = isHalf
        self.model = self.model.half() if isHalf else self.model.float()
