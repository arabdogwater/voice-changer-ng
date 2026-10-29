import torch
from const import PitchExtractorType
from voice_changer.DiffusionSVC.pitchExtractor.rmvpe.rmvpe import RMVPE
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor, analysis_window, coarse_pitch, pitch_scale, place_tail

HOP = 160  # RMVPE is fixed to 16 kHz / 10 ms


class RMVPEPitchExtractor(PitchExtractor):
    gpu_native = True

    def __init__(self, file: str, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "rmvpe"
        dm = DeviceManager.get_instance()
        self.device = dm.getDevice(gpu)
        self.rmvpe = RMVPE(model_path=file, is_half=dm.halfPrecisionAvailable(gpu), device=self.device)
        if self.device.type == "cuda":  # NHWC lets cuDNN use tensor-core kernels for the U-Net (1.6x, identical output)
            self.rmvpe.model = self.rmvpe.model.to(memory_format=torch.channels_last)

    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key, sr, window, silence_front=0):
        # 100 ms of left context for the U-Net, at least 32 frames for its pooling
        start = analysis_window(audio.shape[0], silence_front, 16000, HOP, context=1600, min_len=32 * HOP)
        f0 = self.rmvpe.infer_from_audio_gpu(audio[start:], thred=0.03)
        f0 = f0 * pitch_scale(f0_up_key)
        pitchf = place_tail(pitchf, f0[: (audio.shape[0] - start) // HOP + 1])
        return coarse_pitch(pitchf), pitchf
