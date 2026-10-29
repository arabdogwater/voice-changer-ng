from dataclasses import dataclass, field


@dataclass
class RVCSettings:
    gpu: int = -9999
    dstId: int = 0

    f0Detector: str = "rmvpe"  # rmvpe (gpu, default), fcpe (gpu), crepe*, dio, harvest
    tran: int = 12
    silentThreshold: float = 0.00001
    vad: int = 0  # 1: Silero VAD gate (skips the GPU while nobody speaks)
    vadThreshold: float = 0.4
    extraConvertSize: int = 1024 * 4

    indexRatio: float = 0
    protect: float = 0.5
    rvcQuality: int = 0
    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 48000

    speakers: dict[str, int] = field(default_factory=lambda: {})
    # isHalf: int = 1  # 0:off, 1:on
    # enableDirectML: int = 0  # 0:off, 1:on
    # ↓mutableな物だけ列挙
    intData = [
        "vad",
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "rvcQuality",
        "silenceFront",
    ]
    floatData = ["silentThreshold", "indexRatio", "protect", "vadThreshold"]
    strData = ["f0Detector"]
