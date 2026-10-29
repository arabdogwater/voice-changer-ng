"""Objective comparison of converted audio between two bench runs (e.g. original exe vs this fork).

RVC injects random noise, so outputs are never sample-identical. Instead this compares what a
listener hears: log-mel spectrum distance, pitch track error in cents, and loudness.

    .venv/Scripts/python bench/compare_audio.py --a old_rmvpe --b new_rmvpe
"""

import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from voice_changer.DiffusionSVC.pitchExtractor.rmvpe.rmvpe import RMVPE  # noqa: E402
from voice_changer.utils.mel import mel as mel_basis  # noqa: E402
import soxr  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rmvpe = RMVPE(os.path.join(HERE, "..", "server", "pretrain", "rmvpe.pt"), is_half=False, device=dev)
MEL = torch.from_numpy(mel_basis(48000, 2048, 80, 40, 12000)).to(dev)


def logmel(x):
    spec = torch.stft(torch.from_numpy(x).to(dev), 2048, 480, window=torch.hann_window(2048, device=dev), return_complex=True).abs()
    return torch.log10(MEL @ spec + 1e-5) * 20  # dB


def f0(x):
    x16 = soxr.resample(x, 48000, 16000).astype(np.float32)
    return rmvpe.infer_from_audio_gpu(torch.from_numpy(x16).to(dev)).cpu().numpy()


def load(p):
    x, sr = sf.read(p, dtype="float32")
    assert sr == 48000
    return x


def compare(pa, pb):
    a, b = load(pa), load(pb)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rms_a, rms_b = np.sqrt((a**2).mean()), np.sqrt((b**2).mean())
    # loudness-normalise before spectral comparison (upstream scales output by sqrt(input rms))
    ma, mb = logmel(a / (rms_a + 1e-9)), logmel(b / (rms_b + 1e-9))
    active = (ma.mean(0) > ma.mean(0).max() - 40) & (mb.mean(0) > mb.mean(0).max() - 40)
    mel_db = (ma - mb).abs()[:, active].mean().item()
    fa, fb = f0(a), f0(b)
    m = min(len(fa), len(fb))
    fa, fb = fa[:m], fb[:m]
    voiced = (fa > 0) & (fb > 0)
    cents = np.abs(1200 * np.log2(fa[voiced] / fb[voiced]))
    vuv_agree = ((fa > 0) == (fb > 0)).mean()
    return {
        "mel_dist_db": mel_db,
        "f0_median_cents": float(np.median(cents)) if cents.size else float("nan"),
        "f0_within_50c": float((cents < 50).mean()) if cents.size else float("nan"),
        "vuv_agreement": float(vuv_agree),
        "loudness_ratio_db": float(20 * np.log10((rms_b + 1e-9) / (rms_a + 1e-9))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="label of the reference run")
    ap.add_argument("--b", required=True, help="label of the run to check")
    ap.add_argument("--dir", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    print(f"{'config':>16} | {'mel dB':>7} {'f0 med c':>8} {'f0<50c':>7} {'V/UV':>6} {'loud dB':>8}")
    for pa in sorted(glob.glob(os.path.join(args.dir, f"{args.a}_c*.wav"))):
        cfg = os.path.basename(pa)[len(args.a) + 1 : -4]
        pb = os.path.join(args.dir, f"{args.b}_{cfg}.wav")
        if not os.path.exists(pb) or os.path.getsize(pa) < 10000 or os.path.getsize(pb) < 10000:
            continue
        r = compare(pa, pb)
        print(f"{cfg:>16} | {r['mel_dist_db']:7.2f} {r['f0_median_cents']:8.1f} {r['f0_within_50c']:7.1%} {r['vuv_agreement']:6.1%} {r['loudness_ratio_db']:+8.2f}")


if __name__ == "__main__":
    main()
