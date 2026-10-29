"""Fairseq-free HuBERT / ContentVec (base) for inference.

Loads the original fairseq checkpoints (hubert_base.pt, checkpoint_best_legacy_500.pt,
rinna japanese hubert) without importing fairseq/omegaconf, and runs them with
fused scaled-dot-product attention. Numerically matches fairseq's
`HubertModel.extract_features(source, output_layer=L)` (+ optional `final_proj`).
"""

import os
import pickle
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __call__(self, *args, **kwargs):
        return _Stub()


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("fairseq", "omegaconf", "argparse")):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


_stub_pickle = types.ModuleType("stub_pickle")
_stub_pickle.Unpickler = _StubUnpickler  # type: ignore[attr-defined]
_stub_pickle.load = pickle.load  # type: ignore[attr-defined]


def load_fairseq_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Return the model state dict of a fairseq checkpoint (or of our safetensors cache)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    ckpt = torch.load(path, map_location="cpu", pickle_module=_stub_pickle, weights_only=False)
    return ckpt["model"] if "model" in ckpt else ckpt


class _ConvFeatureExtractor(nn.Module):
    # fairseq default: [(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2, group norm on the first layer
    def __init__(self):
        super().__init__()
        spec = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        self.convs = nn.ModuleList()
        in_ch = 1
        for ch, k, s in spec:
            self.convs.append(nn.Conv1d(in_ch, ch, k, s, bias=False))
            in_ch = ch
        self.norm0 = nn.GroupNorm(512, 512, affine=True)

    def forward(self, x):  # [B, T] -> [B, C, frames]
        x = x.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i == 0:
                x = self.norm0(x)
            x = F.gelu(x)
        return x


class _EncoderLayer(nn.Module):
    def __init__(self, dim=768, heads=12, ffn=3072):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):  # post-LN (layer_norm_first=False)
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, t, c)
        x = self.attn_norm(x + self.out_proj(a))
        x = self.final_norm(x + self.fc2(F.gelu(self.fc1(x))))
        return x


class HubertLite(nn.Module):
    def __init__(self, n_layers=12, dim=768, proj_dim=256):
        super().__init__()
        self.feature_extractor = _ConvFeatureExtractor()
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, dim)
        self.pos_conv = nn.Conv1d(dim, dim, 128, padding=64, groups=16)
        self.enc_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([_EncoderLayer(dim) for _ in range(n_layers)])
        self.final_proj = nn.Linear(dim, proj_dim)

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor, output_layer: int = 12, final_proj: bool = False) -> torch.Tensor:
        """wav: [B, T] float at 16 kHz -> [B, frames, 768|256]"""
        x = self.feature_extractor(wav).transpose(1, 2)
        x = self.post_extract_proj(self.layer_norm(x))
        pc = self.pos_conv(x.transpose(1, 2))[:, :, :-1]  # SamePad for even kernel
        x = self.enc_norm(x + F.gelu(pc).transpose(1, 2))
        for layer in self.layers[:output_layer]:
            x = layer(x)
        return self.final_proj(x) if final_proj else x

    def load_fairseq(self, sd: dict[str, torch.Tensor]):
        new: dict[str, torch.Tensor] = {}
        for i in range(7):
            new[f"feature_extractor.convs.{i}.weight"] = sd[f"feature_extractor.conv_layers.{i}.0.weight"]
        new["feature_extractor.norm0.weight"] = sd["feature_extractor.conv_layers.0.2.weight"]
        new["feature_extractor.norm0.bias"] = sd["feature_extractor.conv_layers.0.2.bias"]
        for k in ["layer_norm.weight", "layer_norm.bias", "post_extract_proj.weight", "post_extract_proj.bias", "final_proj.weight", "final_proj.bias"]:
            new[k] = sd[k]
        # weight norm (dim=2): w = g * v / ||v|| with the norm over every dim except 2
        g = sd["encoder.pos_conv.0.weight_g"].float()
        v = sd["encoder.pos_conv.0.weight_v"].float()
        new["pos_conv.weight"] = g * v / v.norm(dim=(0, 1), keepdim=True)
        new["pos_conv.bias"] = sd["encoder.pos_conv.0.bias"]
        new["enc_norm.weight"] = sd["encoder.layer_norm.weight"]
        new["enc_norm.bias"] = sd["encoder.layer_norm.bias"]
        for i in range(len(self.layers)):
            p = f"encoder.layers.{i}."
            # fairseq scales q by head_dim**-0.5 inside the attention; SDPA does the same, so q is used as-is
            new[f"layers.{i}.qkv.weight"] = torch.cat([sd[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])
            new[f"layers.{i}.qkv.bias"] = torch.cat([sd[p + f"self_attn.{n}_proj.bias"] for n in "qkv"])
            for a, b in [("out_proj", "self_attn.out_proj"), ("attn_norm", "self_attn_layer_norm"), ("fc1", "fc1"), ("fc2", "fc2"), ("final_norm", "final_layer_norm")]:
                new[f"layers.{i}.{a}.weight"] = sd[p + b + ".weight"]
                new[f"layers.{i}.{a}.bias"] = sd[p + b + ".bias"]
        self.load_state_dict({k: t.float() for k, t in new.items()})
        return self


def load_hubert_lite(path: str, device: torch.device, half: bool) -> HubertLite:
    """Load a fairseq HuBERT/ContentVec checkpoint. A safetensors copy is cached next to it for fast reloads."""
    cache = os.path.splitext(path)[0] + ".lite.safetensors"
    sd = load_fairseq_state_dict(cache if os.path.exists(cache) else path)
    model = HubertLite()
    if "pos_conv.weight" in sd:  # already converted
        model.load_state_dict({k: t.float() for k, t in sd.items()})
    else:
        model.load_fairseq(sd)
        try:
            from safetensors.torch import save_file

            save_file({k: t.half().contiguous() for k, t in model.state_dict().items()}, cache)
        except Exception:  # cache is an optimisation only
            pass
    model = model.eval().to(device)
    return model.half() if half else model
