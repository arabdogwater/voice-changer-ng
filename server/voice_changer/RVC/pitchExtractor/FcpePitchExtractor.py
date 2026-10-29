import torch
from const import PitchExtractorType
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor, analysis_window, coarse_pitch, pitch_scale, place_tail

HOP = 160


class FcpePitchExtractor(PitchExtractor):
    gpu_native = True

    def __init__(self, gpu: int):
        super().__init__()
        import torchfcpe

        self.pitchExtractorType: PitchExtractorType = "fcpe"
        self.device = DeviceManager.get_instance().getDevice(gpu)
        self.fcpe = torchfcpe.spawn_bundled_infer_model(self.device)

    @torch.inference_mode()
    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key, sr, window, silence_front=0):
        start = analysis_window(audio.shape[0], silence_front, 16000, HOP, context=1600, min_len=8 * HOP)
        f0 = self.fcpe.infer(
            audio[start:].to(self.device).unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).reshape(-1)
        f0 = f0 * pitch_scale(f0_up_key)
        pitchf = place_tail(pitchf, f0)
        return coarse_pitch(pitchf), pitchf
