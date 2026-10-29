from typing import Any, Protocol
import torch
import onnxruntime

from const import EnumInferenceTypes


class Inferencer(Protocol):
    inferencerType: EnumInferenceTypes = EnumInferenceTypes.pyTorchRVC
    file: str
    isHalf: bool = True
    gpu: int = 0

    model: onnxruntime.InferenceSession | Any | None = None

    def loadModel(self, file: str, gpu: int):
        ...

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def setProps(
        self,
        inferencerType: EnumInferenceTypes,
        file: str,
        isHalf: bool,
        gpu: int,
    ):
        self.inferencerType = inferencerType
        self.file = file
        self.isHalf = isHalf
        self.gpu = gpu

    def getInferencerInfo(self):
        return {
            "inferencerType": self.inferencerType.value,
            "file": self.file,
            "isHalf": self.isHalf,
            "gpu": self.gpu,
        }


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Fold weight norm into plain weights and drop the training-only posterior encoder.

    Upstream kept the weight_norm hooks, so every conv recomputed g * v / ||v|| on every chunk.
    """
    for m in model.modules():
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            torch.nn.utils.remove_weight_norm(m)
    if hasattr(model, "enc_q"):
        del model.enc_q
    return model
