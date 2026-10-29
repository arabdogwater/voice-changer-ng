# voice-changer-ng

[![CI](https://github.com/arabdogwater/voice-changer-ng/actions/workflows/ci.yml/badge.svg)](https://github.com/arabdogwater/voice-changer-ng/actions/workflows/ci.yml)

A next-gen fork of [w-okada/voice-changer](https://github.com/w-okada/voice-changer) (MMVCServerSIO / VC Client) with a rewritten **RVC engine**: same models, same web UI, same API, **4.5–5.6x lower per-chunk latency end to end** and real time at chunk sizes the original can't sustain.

Upstream's open-source v1 line has been frozen since early 2024 (Python 3.10, torch 2.0, fairseq). This fork moves it to Python 3.12 / PyTorch 2.11 / CUDA 12.8 and rebuilds the RVC hot path around the GPU.

## Results (RTX 3070, RVC v2 48 kHz model, RMVPE, measured end-to-end through the REST API)

| chunk | extra | original exe p50 / p95 (ms) | voice-changer-ng p50 / p95 (ms) | speed-up (p50) |
|---|---|---|---|---|
| 1024 (21 ms) | 4096 | – | 13.2 / 30.4 ❌ | – |
| 1024 (21 ms) | 16384 | – | 14.3 / 31.0 ❌ | – |
| 1024 (21 ms) | 131072 | – | 18.8 / 35.2 ❌ | – |
| 2048 (43 ms) | 4096 | 76.4 / 105.6 ❌ | 14.4 / 31.5 ✅ | **5.3x** |
| 2048 (43 ms) | 16384 | 78.0 / 96.4 ❌ | 14.3 / 30.8 ✅ | **5.5x** |
| 2048 (43 ms) | 131072 | 89.4 / 126.1 ❌ | 19.8 / 40.8 ✅ | **4.5x** |
| 4096 (85 ms) | 4096 | 81.5 / 148.4 ❌ | 14.6 / 37.3 ✅ | **5.6x** |
| 4096 (85 ms) | 16384 | 76.5 / 97.9 ❌ | 14.6 / 34.2 ✅ | **5.2x** |
| 4096 (85 ms) | 131072 | 95.7 / 125.7 ❌ | 19.5 / 36.8 ✅ | **4.9x** |
| 8192 (171 ms) | 4096 | 79.1 / 101.0 ✅ | 15.9 / 18.9 ✅ | **5.0x** |
| 8192 (171 ms) | 16384 | 76.5 / 89.2 ✅ | 16.0 / 23.4 ✅ | **4.8x** |
| 8192 (171 ms) | 131072 | 95.4 / 117.2 ✅ | 20.5 / 40.3 ✅ | **4.7x** |

✅ = p95 below the chunk duration (keeps up in real time), ❌ = falls behind. The original exe has a ~76 ms floor per chunk no matter how small the chunk is, so it never keeps up below ~170 ms chunks; 1024-sample chunks were not measured on it (they cannot work with that floor).

**TensorRT (`--tensorrt true`)** on top, measured as an interleaved A/B in one process so both variants see the same background GPU load: **12.6 → 10.6 ms p50 per chunk** (≈ −2 ms, ~16 %), and in a server run 10.7 / 14.2 ms p50 / p95 at 1024-sample (21 ms) chunks. Audio vs the PyTorch decoder: 1.6–1.8 dB mel distance and 3–4 cents, i.e. inside the run-to-run noise floor.

Numbers include ~4 ms of HTTP/JSON overhead per chunk (the `/test` REST endpoint both servers share); the pipeline itself takes ~10–12 ms in-process. They were taken on a desktop GPU that was also driving the display and other apps, so expect a few ms of run-to-run drift.

Same model, same input, same benchmark client (`bench/bench_rest.py`) against the original `MMVCServerSIO.exe` and this fork. "p95" is the 95th percentile of the per-chunk round trip; a chunk is real-time when that is below the chunk duration.

**Audio quality is unchanged.** RVC injects random noise, so outputs are never bit-identical; `bench/compare_audio.py` compares what you hear instead:

| old exe vs this fork | value | run-to-run noise floor (fork vs itself) |
|---|---|---|
| log-mel distance | 2.0 – 3.4 dB | 1.3 – 2.0 dB |
| median pitch difference | 4.4 – 6.8 cents (inaudible; a semitone is 100) | 2.3 – 5.4 cents |
| voiced/unvoiced agreement | 94 – 99.7 % | 96 – 100 % |
| loudness difference | within ±0.2 dB (one config −1.3 dB) | within ±0.07 dB |

## What changed

| Area | Upstream v1 | voice-changer-ng | Why it matters |
|---|---|---|---|
| Runtime | Python 3.10, torch 2.0.1 + CUDA 11.8, cuDNN 8 | Python 3.12, torch 2.11 + CUDA 12.8, cuDNN 9, `uv` lockfile | current drivers/kernels, reproducible installs |
| Kernel launches | ~2,900 small CUDA kernels launched from Python per chunk (CPU-bound: ~70 ms CPU for ~20 ms of GPU work) | whole chunk captured once as a **CUDA graph**, replayed with one launch | the main speed-up |
| HuBERT / ContentVec | fairseq (or an fp32 ONNX export) | `HubertLite`: pure PyTorch, fused SDPA attention, fp16; loads the same `hubert_base.pt` | drops fairseq/hydra/omegaconf; output matches the ONNX reference to cosine 1.00000 |
| Synthesizer | text encoder, flow, NSF sine source and decoder run over chunk + full history (`extraConvertSize`) | only the text encoder sees the whole window; flow, source and decoder run on the exact receptive field the output needs | a big `extraConvertSize` costs ~1 ms instead of a full re-synthesis; proven equal to full synthesis (tests) |
| Weight norm | recomputed `g·v/‖v‖` for every conv on every chunk | folded into the weights at load; training-only posterior encoder dropped | fewer kernels, less VRAM |
| RMVPE | fp32, frame-by-frame Python decode loop, crashes on short clips with `silenceFront` | fp16, channels-last U-Net (5.7 → 3.5 ms), vectorised GPU decode, 100 ms left context | faster, and fixes the upstream bug where chunks ≤ 4096 samples returned 1 sample |
| Index (`.index`) | CPU faiss IVF search with GPU→CPU→GPU copies | exact nearest-neighbour search on the GPU | faster *and* more accurate than nprobe=1 IVF |
| Buffers / resampling | numpy ring buffers, `resampy` on the CPU | GPU ring buffers, GPU resampling | 2 small host↔device copies per chunk |
| Pitch + features | sequential | two CUDA streams inside the graph | the two latency-bound branches overlap |
| TensorRT | – | optional fp16 decoder engines (`--tensorrt true`), pre-built once per voice + chunk size, cached | decoder 4.15 → 1.75 ms, whole chunk ≈ −2 ms |
| Silence | RMS threshold only | optional Silero VAD gate (`--vad true`) | zero GPU work while nobody speaks, without letting fan/keyboard noise through |
| Weights | downloads every model family's weights at boot (~2 GB); saved HTTP error pages as model files | core weights only, the rest on first use; atomic `.part` downloads, HTTP errors raise | faster first start, no corrupt "models" |
| Model loading | `torch.load` (pickle, can run code) | PyTorch's `weights_only` loader (all 52 tested community `.pth` files load) | a random model from the internet can't execute code |
| Desktop client | 163 MB Electron app; the server quits when its window closes | the same web UI opens in your browser (`--launch_client native` still works) | lighter, no hidden coupling |
| Server audio | sounddevice default `latency="high"` | `latency="low"` (WASAPI exclusive still available) | less host buffering in server-audio mode |

## Quick start (Windows, NVIDIA)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
2. Clone and start:

   ```bash
   git clone https://github.com/arabdogwater/voice-changer-ng.git
   ```

   ```bash
   cd voice-changer-ng && start_ng.bat
   ```

   The first run creates `.venv` (Python 3.12 + CUDA PyTorch, a few GB) and downloads `hubert_base.pt` + `rmvpe.pt`. The UI opens at http://localhost:18888.

3. Already have the original VC Client with your voices? Point the fork at its model folder, nothing needs converting:

   ```bash
   start_ng.bat --model_dir D:\AI\MMVCServerSIO\model_dir
   ```

Linux: `./start_ng.sh` (same arguments).

### Options

| Flag | Default | |
|---|---|---|
| `--tensorrt true` | off | TensorRT fp16 decoder. Install with `uv sync --extra tensorrt`, then pre-build the engines for the voices and chunk size you use (below). Missing engines are built in the background on first use while audio keeps running on the PyTorch path, but that competes with the voice changer for the GPU for several minutes. |
| `--vad true` | off | Silero VAD gate. Opens on speech, closes ~400 ms after it; the first ~40 ms of a phrase can be gated, tune with the `vadThreshold` setting (0.4). |
| `--launch_client` | `browser` | `browser`, `native` (legacy Electron client, if present) or `none`. |
| `--download_all_weights true` | off | pre-fetch every weight (crepe, whisper, DDSP/Diffusion vocoders…) at boot. |

### TensorRT engines

Engines are specific to a voice model and to the chunk / crossfade / extra values set in the UI. Build them once with nothing else using the GPU (~3 min each), and they are cached in `server/engine_cache/`:

```bash
uv sync --extra tensorrt
```

```bash
cd server && uv run --no-sync python tools/build_trt_engines.py --model_dir model_dir --slots 0,3 --chunks 2048 --crossfade 4096 --extra 16384
```

### Recommended settings

- **Chunk**: 2048 samples (43 ms) is comfortably real-time on an RTX 3070; 1024 (21 ms) works with TensorRT on.
- **Extra (`extraConvertSize`)**: 16384–32768 gives the text encoder enough context; larger values now cost very little.
- **F0**: `rmvpe` (GPU, graph-captured). `fcpe` also runs on the GPU but cannot be graph-captured, so it is slower here.
- **NVIDIA Control Panel → Power management mode → Prefer maximum performance** for `python.exe`: the GPU otherwise downclocks between chunks, which costs a few ms.

## What is and isn't optimised

The rewrite covers the **RVC** path end to end (all RVC v1/v2, 32/40/48 kHz, f0 and no-f0, WebUI-style models). The other model families from upstream (MMVC v1.3/v1.5, so-vits-svc 4.0, DDSP-SVC, Diffusion-SVC, Beatrice, LLVC, EasyVC) are kept as upstream code. They need `uv sync --extra legacy` and are not benchmarked here. Beatrice ships as a closed Python 3.10 binary, so it does not load on Python 3.12.

## Development

```bash
uv sync --extra dev
```

```bash
uv run --no-sync pytest -q
```

- `bench/bench_rest.py`: streaming benchmark against any running server (old or new).
- `bench/profile_pipeline.py`: in-process profiler (`--stages` for a per-stage breakdown in eager mode).
- `bench/compare_audio.py`: objective quality comparison between two runs.
- `general_instructions.md`: architecture notes and the rules the hot path must follow (no host syncs inside the captured region).

## Roadmap

Measured, not done yet, in order of expected value:

- TensorRT for RMVPE and HuBERT (both are already graph-captured; expected ~2–3 ms more per chunk).
- Refittable TensorRT engines: one engine per architecture + chunk size, weights refitted per voice in seconds instead of a full build.
- Optional neural denoiser in front of the embedder (DeepFilterNet / RNNoise).
- Newer streaming VC families (e.g. zero-shot models) as additional engines.

Considered and dropped: a Rust audio loop (with CUDA graphs the Python overhead per chunk is ~1 ms, so it would not move the needle), and GPU SOLA (the CPU crossfade takes ~0.3 ms).

## Credits and license

MIT, same as upstream. This is a fork of [w-okada/voice-changer](https://github.com/w-okada/voice-changer) by Wataru Okada; all upstream licenses and notices (`LICENSE`, `LICENSE-NOTICE`, `LICENSE-CLA`) are kept. Built on [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), [ContentVec](https://github.com/auspicious3000/contentvec), [RMVPE](https://github.com/Dream-High/RMVPE), [FCPE](https://github.com/CNChTu/FCPE), [Silero VAD](https://github.com/snakers4/silero-vad) and [TensorRT](https://developer.nvidia.com/tensorrt). Upstream's original Japanese README is in `README_upstream_ja.md`, its English one in `README_en.md`.
