import os
from concurrent.futures import ThreadPoolExecutor

from downloader.Downloader import download
from mods.log_control import VoiceChangaerLogger
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from Exceptions import WeightDownladException

logger = VoiceChangaerLogger.get_instance().getLogger()

HF = "https://huggingface.co"

# key -> (VoiceChangerParams attribute or callable(params) -> path, url)
WEIGHTS = {
    "hubert_base": ("hubert_base", f"{HF}/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt"),
    "hubert_base_jp": ("hubert_base_jp", f"{HF}/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt"),
    "hubert_soft": ("hubert_soft", f"{HF}/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt"),
    "nsf_hifigan": ("nsf_hifigan", f"{HF}/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin"),
    "nsf_hifigan_config": (lambda p: os.path.join(os.path.dirname(p.nsf_hifigan), "config.json"), f"{HF}/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json"),
    "nsf_hifigan_onnx": (lambda p: os.path.join(os.path.dirname(p.nsf_hifigan), "nsf_hifigan.onnx"), f"{HF}/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_onnx_20221211/nsf_hifigan.onnx"),
    "crepe_onnx_full": ("crepe_onnx_full", f"{HF}/wok000/weights/resolve/main/crepe/onnx/full.onnx"),
    "crepe_onnx_tiny": ("crepe_onnx_tiny", f"{HF}/wok000/weights/resolve/main/crepe/onnx/tiny.onnx"),
    "content_vec_500_onnx": ("content_vec_500_onnx", f"{HF}/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx"),
    "rmvpe": ("rmvpe", f"{HF}/wok000/weights/resolve/main/rmvpe/rmvpe_20231006.pt"),
    "rmvpe_onnx": ("rmvpe_onnx", f"{HF}/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx"),
    "silero_vad": (lambda p: os.path.join(os.path.dirname(p.rmvpe), "silero_vad.onnx"), "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"),
    "whisper_tiny": ("whisper_tiny", "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"),
}

# what an RVC session needs; everything else is fetched the first time a feature asks for it
CORE_WEIGHTS = ["hubert_base", "rmvpe"]
# DDSP-SVC / Diffusion-SVC / so-vits use these
LEGACY_MODEL_WEIGHTS = ["hubert_base", "hubert_soft", "nsf_hifigan", "nsf_hifigan_config", "nsf_hifigan_onnx", "content_vec_500_onnx", "rmvpe", "crepe_onnx_full", "crepe_onnx_tiny"]


def weightPath(params: VoiceChangerParams, key: str) -> str:
    target, _ = WEIGHTS[key]
    return target(params) if callable(target) else getattr(params, target)


def ensureWeights(params: VoiceChangerParams, keys: list[str]):
    """Download the listed weights if missing. Raises WeightDownladException if any is still missing."""
    jobs = []
    for i, key in enumerate(keys):
        path = weightPath(params, key)
        if not os.path.exists(path):
            jobs.append({"url": WEIGHTS[key][1], "saveTo": path, "position": i})
    if jobs:
        with ThreadPoolExecutor() as pool:
            list(pool.map(download, jobs))
    missing = [weightPath(params, k) for k in keys if not os.path.exists(weightPath(params, k))]
    for k in keys:
        p = weightPath(params, k)
        if os.path.exists(p):
            logger.debug(f"weight file [{p}]: {os.path.getsize(p)}")
    if missing:
        logger.warning(f"weight files are missing: {missing}")
        raise WeightDownladException()


def downloadWeight(voiceChangerParams: VoiceChangerParams, all_weights: bool = False):
    ensureWeights(voiceChangerParams, list(WEIGHTS.keys()) if all_weights else CORE_WEIGHTS)
