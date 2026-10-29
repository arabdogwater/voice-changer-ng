import os
import sys

import pytest
import torch

SERVER = os.path.join(os.path.dirname(__file__), "..", "server")
sys.path.insert(0, os.path.abspath(SERVER))

# exactness tests compare fp32 results; TF32 would add ~1e-3 noise on Ampere+
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@pytest.fixture
def no_noise(monkeypatch):
    """RVC injects random noise (z_p sampling, NSF source); zero it so outputs are deterministic."""
    monkeypatch.setattr(torch, "randn_like", lambda x, **k: torch.zeros_like(x))
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.zeros(*a, **{n: v for n, v in k.items() if n in ("device", "dtype")}))


def pretrain(name: str) -> str:
    path = os.path.join(SERVER, "pretrain", name)
    if not os.path.exists(path):
        pytest.skip(f"needs server/pretrain/{name}")
    return path
