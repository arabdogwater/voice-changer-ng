"""Streaming benchmark for a running voice-changer server (old or new).

Streams a wav file to the server's REST `/test` endpoint in fixed-size chunks,
exactly like the browser client does, and measures the per-chunk round trip.
Because both the original MMVCServerSIO.exe and this fork expose the same
endpoint, the numbers are directly comparable.

    python bench/bench_rest.py --url http://127.0.0.1:18888 --slot 0 \
        --chunks 2048,4096,8192 --extra 4096,16384,131072 --f0 rmvpe --out bench/out/old

Output: a table per config (p50/p95/max ms, real-time factor) plus a JSON file
and the converted audio for every config, used later by bench/compare_audio.py.
"""

import argparse
import base64
import json
import os
import time
import urllib.parse
import urllib.request
import wave

import numpy as np

SR = 48000


def load_wav_48k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    assert width == 2, "16-bit PCM only"
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        t_old = np.arange(len(x)) / sr
        t_new = np.arange(int(len(x) * SR / sr)) / SR
        x = np.interp(t_new, t_old, x)
    return x.astype(np.int16)


def write_wav(path: str, x: np.ndarray):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype(np.int16).tobytes())


def post_form(url: str, data: dict):
    body = urllib.parse.urlencode({k: str(v) for k, v in data.items()}).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=120) as r:
        return json.loads(r.read())


def set_setting(base: str, key: str, val):
    post_form(f"{base}/update_settings", {"key": key, "val": val})


def convert_chunk(base: str, chunk: np.ndarray) -> np.ndarray:
    payload = json.dumps({"timestamp": 0, "buffer": base64.b64encode(chunk.tobytes()).decode()}).encode()
    req = urllib.request.Request(f"{base}/test", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read())
    if isinstance(res, str):
        raise RuntimeError(f"server error: {res}")
    return np.frombuffer(base64.b64decode(res["changedVoiceBase64"]), dtype=np.int16)


def run_config(base: str, audio: np.ndarray, chunk: int, warmup: int, loops: int):
    times, outs, bad = [], {}, 0
    n_chunks = len(audio) // chunk
    total = warmup + n_chunks * loops
    for i in range(total):
        c = audio[(i % n_chunks) * chunk:(i % n_chunks + 1) * chunk]
        t0 = time.perf_counter()
        out = convert_chunk(base, c)
        dt = time.perf_counter() - t0
        bad += len(out) != len(c)
        if i >= warmup:
            times.append(dt * 1000)
            outs.setdefault(i % n_chunks, out)  # keyed by input position: saved audio always starts at the file start
    audio_out = np.concatenate([outs[k] for k in sorted(outs)]) if outs else np.zeros(0, np.int16)
    return np.array(times), audio_out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18888")
    ap.add_argument("--wav", default=os.path.join(os.path.dirname(__file__), "..", "server", "test.wav"))
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--f0", default="rmvpe")
    ap.add_argument("--chunks", default="2048,4096,8192")
    ap.add_argument("--extra", default="4096,16384,131072")
    ap.add_argument("--crossfade", type=int, default=1024)
    ap.add_argument("--silence-front", type=int, default=0)
    ap.add_argument("--index-ratio", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    audio = load_wav_48k_mono(args.wav)

    set_setting(args.url, "modelSlotIndex", args.slot)
    time.sleep(2)
    set_setting(args.url, "gpu", 0)
    set_setting(args.url, "f0Detector", args.f0)
    set_setting(args.url, "silentThreshold", 0)
    set_setting(args.url, "crossFadeOverlapSize", args.crossfade)
    set_setting(args.url, "silenceFront", args.silence_front)
    set_setting(args.url, "indexRatio", args.index_ratio)

    results = []
    print(f"{'chunk':>6} {'ms':>6} {'extra':>7} | {'p50':>7} {'p95':>7} {'max':>7} | {'RTF':>5} {'bad':>4}")
    for extra in [int(x) for x in args.extra.split(",")]:
        set_setting(args.url, "extraConvertSize", extra)
        for chunk in [int(x) for x in args.chunks.split(",")]:
            times, out, bad = run_config(args.url, audio, chunk, args.warmup, args.loops)
            chunk_ms = chunk / SR * 1000
            p50, p95, mx = np.percentile(times, 50), np.percentile(times, 95), times.max()
            rtf = p95 / chunk_ms
            print(f"{chunk:>6} {chunk_ms:>6.1f} {extra:>7} | {p50:>7.1f} {p95:>7.1f} {mx:>7.1f} | {rtf:>5.2f} {bad:>4}")
            write_wav(os.path.join(args.out, f"{args.label}_c{chunk}_e{extra}.wav"), out)
            results.append({"chunk": chunk, "chunk_ms": chunk_ms, "extra": extra, "p50": p50, "p95": p95, "max": float(mx), "rtf": rtf, "bad_chunks": bad})

    with open(os.path.join(args.out, f"{args.label}.json"), "w") as f:
        json.dump({"label": args.label, "f0": args.f0, "slot": args.slot, "silence_front": args.silence_front, "index_ratio": args.index_ratio, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
