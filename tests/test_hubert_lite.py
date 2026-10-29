"""HubertLite must reproduce fairseq HuBERT. Without the weights we check the parameter mapping;
with server/pretrain present we check the output against the ONNX ContentVec the old build used."""

import numpy as np
import pytest
import torch

from conftest import pretrain
from voice_changer.RVC.embedder.hubert_lite import HubertLite


def _fake_fairseq_state_dict(model: HubertLite) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    sd = {}
    for i, conv in enumerate(model.feature_extractor.convs):
        sd[f"feature_extractor.conv_layers.{i}.0.weight"] = torch.randn_like(conv.weight)
    sd["feature_extractor.conv_layers.0.2.weight"] = torch.randn(512)
    sd["feature_extractor.conv_layers.0.2.bias"] = torch.randn(512)
    for k in ["layer_norm.weight", "layer_norm.bias", "post_extract_proj.weight", "post_extract_proj.bias", "final_proj.weight", "final_proj.bias"]:
        sd[k] = torch.randn_like(model.state_dict()[k])
    sd["encoder.pos_conv.0.weight_g"] = torch.rand(1, 1, 128) + 0.5
    sd["encoder.pos_conv.0.weight_v"] = torch.randn(768, 48, 128)
    sd["encoder.pos_conv.0.bias"] = torch.randn(768)
    sd["encoder.layer_norm.weight"] = torch.randn(768)
    sd["encoder.layer_norm.bias"] = torch.randn(768)
    for i in range(12):
        p = f"encoder.layers.{i}."
        for n in "qkv":
            sd[p + f"self_attn.{n}_proj.weight"] = torch.randn(768, 768)
            sd[p + f"self_attn.{n}_proj.bias"] = torch.randn(768)
        for n, shape in [("self_attn.out_proj", (768, 768)), ("fc1", (3072, 768)), ("fc2", (768, 3072))]:
            sd[p + n + ".weight"] = torch.randn(*shape)
            sd[p + n + ".bias"] = torch.randn(shape[0])
        for n in ["self_attn_layer_norm", "final_layer_norm"]:
            sd[p + n + ".weight"] = torch.randn(768)
            sd[p + n + ".bias"] = torch.randn(768)
    return sd


def test_fairseq_mapping():
    m = HubertLite()
    sd = _fake_fairseq_state_dict(m)
    m.load_fairseq(sd)
    own = m.state_dict()
    # fused qkv keeps fairseq's q, k, v order
    torch.testing.assert_close(own["layers.3.qkv.weight"][768:1536], sd["encoder.layers.3.self_attn.k_proj.weight"])
    # weight norm (dim=2) folded: every kernel tap is rescaled to norm g
    w = own["pos_conv.weight"]
    torch.testing.assert_close(w.norm(dim=(0, 1)), sd["encoder.pos_conv.0.weight_g"].reshape(-1), rtol=1e-4, atol=1e-4)
    out = m(torch.randn(1, 16000), output_layer=9, final_proj=True)
    assert out.shape == (1, 49, 256)


def test_matches_onnx_contentvec():
    ort = pytest.importorskip("onnxruntime")
    hubert = pretrain("hubert_base.pt")
    onnx = pretrain("content_vec_500.onnx")
    from voice_changer.RVC.embedder.hubert_lite import load_hubert_lite

    rng = np.random.default_rng(0)
    t = np.arange(32000) / 16000
    wav = (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t) + 0.02 * rng.standard_normal(32000)).astype(np.float32)
    u9, u12, _ = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]).run(None, {"audio": wav[None]})
    m = load_hubert_lite(hubert, torch.device("cpu"), half=False)
    a9 = m(torch.from_numpy(wav)[None], 9, True).numpy()
    a12 = m(torch.from_numpy(wav)[None], 12, False).numpy()
    np.testing.assert_allclose(a9, u9[:, : a9.shape[1]], atol=2e-2)
    np.testing.assert_allclose(a12, u12[:, : a12.shape[1]], atol=2e-2)
