"""Render the README benchmark table from bench/out/*.json (old exe vs fork, optionally + TensorRT)."""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(label):
    path = os.path.join(HERE, "out", f"{label}.json")
    if not os.path.exists(path):
        return {}
    return {(r["chunk"], r["extra"]): r for r in json.load(open(path))["results"]}


def cell(r, chunk_ms):
    if r is None:
        return "–"
    ok = "✅" if r["p95"] < chunk_ms else "❌"
    return f"{r['p50']:.1f} / {r['p95']:.1f} {ok}"


def main():
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="old_rmvpe")
    ap.add_argument("--new", default="final_rmvpe")
    ap.add_argument("--trt", default="final_rmvpe_trt")
    args = ap.parse_args()
    old, new, trt = load(args.old), load(args.new), load(args.trt)
    keys = sorted(set(old) | set(new) | set(trt))
    trt_col = bool(trt)
    print("| chunk | extra | original exe p50 / p95 (ms) | voice-changer-ng p50 / p95 (ms) |" + (" ng + TensorRT |" if trt_col else "") + " speed-up (p50) |")
    print("|---|---|---|---|" + ("---|" if trt_col else "") + "---|")
    for chunk, extra in keys:
        ms = chunk / 48
        o, n, t = old.get((chunk, extra)), new.get((chunk, extra)), trt.get((chunk, extra))
        best = min([x["p50"] for x in (n, t) if x is not None], default=None)
        speed = f"**{o['p50'] / best:.1f}x**" if o and best else "–"
        trt_cell = f" {cell(t, ms)} |" if trt_col else ""
        print(f"| {chunk} ({ms:.0f} ms) | {extra} | {cell(o, ms)} | {cell(n, ms)} |{trt_cell} {speed} |")


if __name__ == "__main__":
    main()
