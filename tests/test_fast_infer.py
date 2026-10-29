"""Cropped streaming synthesis must equal full-window synthesis on the samples it returns."""

import pytest
import torch

from voice_changer.RVC.inferencer.rvc_models.infer_pack.fast_infer import branch_context, keep_frames
from voice_changer.RVC.inferencer.rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid
from voice_changer.RVC.inferencer.Inferencer import optimize_for_inference

# RVC v2 48k / 40k configs (as stored in the .pth "config" field)
CFG_48K = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000]
CFG_40K = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]


def test_branch_context():
    # ResBlock1 branch, k=11, d=(1,3,5): dilated convs 5*(1+3+5)=45 plus three k=11 convs 3*5=15
    assert branch_context(11, [1, 3, 5], two_conv=True) == 60
    assert branch_context(3, [1, 3], two_conv=False) == 4


def _reference(m, phone, pitch, f0, lens, sid):
    g = m.emb_g(sid).unsqueeze(-1)
    m_p, _, xm = m.enc_p(phone, pitch, lens)
    z = m.flow(m_p * xm, xm, g=g, reverse=True)
    return m.dec(z * xm, f0, g=g)[0, 0] if f0 is not None else m.dec(z * xm, g=g)[0, 0]


@pytest.mark.parametrize("cfg", [CFG_48K, CFG_40K], ids=["48k", "40k"])
@pytest.mark.parametrize("convert_length", [4800, 20480])
def test_cropped_equals_full(no_noise, cfg, convert_length):
    torch.manual_seed(0)
    m = optimize_for_inference(SynthesizerTrnMs768NSFsid(*cfg, is_half=False).eval())
    T = 400
    phone = torch.randn(1, T, 768)
    pitch = torch.zeros(1, T, dtype=torch.long)
    f0 = torch.zeros(1, T)  # unvoiced: the NSF sine phase would otherwise depend on the window start
    lens = torch.tensor([T])
    sid = torch.tensor([0])
    with torch.inference_mode():
        ref = _reference(m, phone, pitch, f0, lens, sid)
        out = m.infer(phone, lens, pitch, f0, sid, convert_length=convert_length)[0][0, 0]
    assert keep_frames(m.dec, convert_length) < T  # the crop actually happened
    assert out.shape[0] >= convert_length
    torch.testing.assert_close(out[-convert_length:], ref[-convert_length:], atol=1e-4, rtol=0)

