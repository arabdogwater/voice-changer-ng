"""Streaming-friendly synthesis for the RVC synthesizers.

The text encoder needs the whole window (it is a global-attention transformer),
but the flow, the NSF sine source and the HiFi-GAN decoder only have a short
receptive field. Upstream ran all of them over the full window (chunk + history),
so a 2.7 s `extraConvertSize` meant synthesizing 2.7 s of audio for every 40 ms
chunk. Here everything after the text encoder only sees the frames that can
influence the requested output.
"""

import numpy as np
import torch

# ResidualCouplingBlock: 4 flows x WN(3 layers, kernel 5, dilation 1) -> 4 * 3 * 2 = 24 frames per side.
FLOW_CONTEXT = 24
SAFETY = 8


def branch_context(kernel: int, dilations, two_conv: bool) -> int:
    """One-sided receptive field (in samples of that stage) of a ResBlock branch."""
    half = (kernel - 1) // 2
    return sum(d * half for d in dilations) + (len(dilations) * half if two_conv else 0)


def keep_frames(dec, convert_length: int) -> int:
    """Encoder frames the flow + decoder need to produce the last `convert_length` samples."""
    upp = int(np.prod(dec.upsample_rates))
    frames = -(-convert_length // upp) + dec.ups_size[0] + 3  # conv_pre kernel 7
    noise = max(getattr(dec, "noise_conv_size", [0]))
    frames = max(frames, -(-(convert_length + noise) // upp) + 1)
    return frames + FLOW_CONTEXT + SAFETY


def infer_cropped(model, phone, phone_lengths, pitch, nsff0, sid, convert_length=None):
    g = model.emb_g(sid).unsqueeze(-1)
    m_p, logs_p, x_mask = model.enc_p(phone, pitch, phone_lengths)
    dec = model.dec
    if convert_length is not None and getattr(dec, "realtime", False):
        keep = keep_frames(dec, int(convert_length))
        if keep < m_p.shape[2]:
            m_p, logs_p, x_mask = m_p[:, :, -keep:], logs_p[:, :, -keep:], x_mask[:, :, -keep:]
            if nsff0 is not None:
                nsff0 = nsff0[:, -keep:]
    z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
    z = model.flow(z_p, x_mask, g=g, reverse=True)
    trt = getattr(model, "trt_decoder", None)
    if trt is not None and nsff0 is not None and convert_length is not None and getattr(dec, "realtime", False):
        x = (z * x_mask).float()
        har = dec.m_source(nsff0, dec.upp)[0].transpose(1, 2).float()
        engine = trt.get(x, har, g.float(), int(convert_length))
        if engine is not None:
            return engine(x=x, har=har, g=g.float()), x_mask, None
    if nsff0 is None:
        o = dec.infer_realtime(z * x_mask, g=g, convert_length=convert_length)
    else:
        o = dec.infer_realtime(z * x_mask, nsff0, g=g, convert_length=convert_length)
    return o, x_mask, None
