# voice-changer-ng — project knowledge

How to read: read CORE (down to `END CORE`) for every task, then use the Index to load only the categories the task touches (`Grep "^#{1,2} "` for line numbers). Grep the whole file for a system's name before building something new.

## Index

| Category | Sections |
|---|---|
| Engine | Pipeline flow, CUDA graphs, Cropped synthesis, HubertLite, Pitch extractors, GPU index |
| Acceleration | TensorRT decoder |
| Runtime | Environment, Weights, VAD gate, Server launch |
| Verification | Benchmarks, Quality check, Tests |

# CORE

- What: fork of w-okada/voice-changer (MIT). The RVC path was rewritten for speed; other model families (MMVC, so-vits, DDSP, Diffusion, Beatrice, LLVC, EasyVC) are upstream code, kept importable but not optimised.
- Env: `uv sync` (Python 3.12, torch 2.11+cu128). Caches/temp on D: (`UV_CACHE_DIR=D:\uv-cache`, `UV_PYTHON_INSTALL_DIR=D:\uv-python`); C: is nearly full.
- Run: `start_ng.bat` or `cd server && uv run --no-sync python MMVCServerSIO.py -p 18888 --https false`. UI opens in the browser (no Electron).
- Hard rule: the per-chunk path (`Pipeline._core`) must stay free of host syncs (`.item()`, `.cpu()`, `if tensor`, `torch.tensor(list, device=cuda)`), or CUDA-graph capture breaks. Decide shapes/branches in `Pipeline.exec` (the "plan").
- Hard rule: anything a user can drag (pitch, index ratio, protect, speaker) is a tensor in `knobs`, never part of the graph key.
- Hard rule: never commit weights/models/engines (`server/pretrain`, `server/model_dir`, `server/engine_cache`, `bench/out/*.wav` are ignored).
- Verify every engine change with `bench/compare_audio.py` (quality) and `bench/bench_rest.py` / `bench/profile_pipeline.py` (speed), plus `pytest`.
- The original exe for A/B baselines lives at `D:\AI\MMVCServerSIO` (back up its `stored_setting.json` before benchmarking it; it launches an Electron window and exits when it closes).

END CORE

# Engine

## Pipeline flow
`RVCr2.inference` (int16 @ input SR) → host RMS gate → GPU resample to 16 k (`torchaudio.transforms.Resample`, cached) → `generate_input` (GPU ring buffers for audio / pitchf / features, trimmed to `convertSize`) → `Pipeline.exec` → output × 32767.5·sqrt(rms) (upstream loudness behaviour) → GPU resample to output SR → numpy → `VoiceChangerV2` SOLA crossfade.

## CUDA graphs
`server/voice_changer/RVC/pipeline/cuda_graph.py` `GraphCache`: key = plan tuple, LRU of 4, 2 warm-up runs on a side stream, capture with `capture_error_mode="thread_local"` (background threads may use the GPU). Outputs are cloned after replay. Any capture failure disables graphs for that pipeline and falls back to eager. FCPE is not capturable (host prints in its mel module), so it runs eager.

## Cropped synthesis
`infer_pack/fast_infer.py`: the text encoder sees the whole window; flow + NSF source + decoder only see `keep_frames(dec, convert_length)` frames. Exact (tested to 1e-4 with TF32 off). `branch_context` gives the exact ResBlock receptive field; upstream's formula under-counted.

## HubertLite
`embedder/hubert_lite.py`: fairseq-free HuBERT/ContentVec base, loads fairseq checkpoints through a stub unpickler, fused SDPA attention, caches `*.lite.safetensors` next to the checkpoint. Matches the ONNX ContentVec the old build used (cos 1.00000).

## Pitch extractors
GPU-native extractors (`gpu_native = True`: RMVPE, FCPE) take/return tensors; CPU ones go through `legacy_extract`. RMVPE: fp16 + channels_last U-Net + vectorised GPU decode; analysis keeps 100 ms left context and ≥32 frames (upstream crashed on short clips with silenceFront). `rmvpe_onnx` maps to the torch RMVPE.

## GPU index
`pipeline/gpu_index.py`: faiss only reads the `.index` once (`reconstruct_n`), search is exact L2 on the GPU in chunks (≤32 M scores per matmul).

# Acceleration

## TensorRT decoder
Opt-in `--tensorrt true` (needs `uv sync --extra tensorrt`). `inferencer/trt_decoder.py`: `DecoderCore` = NSF decoder without the random source; ONNX export + fp16 engine at builder optimisation level 2, built in a background thread on first use of (model, decoder shape, convert_length), cached in `server/engine_cache/`. `TrtDecoderCache.generation` is part of the graph key, so the pipeline re-captures once an engine lands. Builds never start inside a capture. Decoder 4.15 → 1.75 ms (RTX 3070); whole chunk 10.9 → 9.4 ms.
Engines depend on (model, chunk, crossfade, extra rounding). Building while audio runs competes for the GPU (took >5 min and raised latency to ~25 ms p50 in testing), so pre-build offline: `server/tools/build_trt_engines.py --slots 0 --chunks 2048 --crossfade <UI value> --extra <UI value>` (~3 min per engine, nothing else on the GPU).

# Runtime

## Environment
`pyproject.toml` + `uv.lock`; torch/torchaudio from the cu128 index. Extras: `legacy` (pyworld, torchcrepe, librosa… for CPU pitch and non-RVC families), `tensorrt`, `dev`. No fairseq / librosa / resampy at runtime: `utils/mel.py` replaces `librosa.filters.mel`, `utils/resample.py` is a soxr-backed `resampy.resample`.

## Weights
`downloader/WeightDownloader.py`: registry `WEIGHTS`; boot fetches `CORE_WEIGHTS` (hubert_base, rmvpe) only, everything else via `ensureWeights` on first use (`--download_all_weights true` for all). Downloads go to `.part` and are renamed; HTTP errors raise (upstream saved error pages as models).

## VAD gate
`utils/vad.py` `SileroVAD` (CPU ONNX, 512-sample windows + 64 context, state carried across chunks, 400 ms hangover). `RVCr2._vad_open` gates before any GPU work; weight `silero_vad` is fetched on first use. Settings `vad` (int) / `vadThreshold` (0.4), CLI default `--vad`. Opens on 100 % of loud chunks of `test.wav`; can gate the first chunk of a phrase.

## Server launch
`--launch_client browser|native|none` (default browser). Server audio mode uses sounddevice with `latency=("low","low")`; WASAPI exclusive is the `exclusiveMode` setting.

# Verification

## Benchmarks
`bench/bench_rest.py --url … --label X` streams `server/test.wav` through `/test` in fixed chunks for chunk × extraConvertSize configs, writes `bench/out/X.json` + wavs. `bench/profile_pipeline.py` is the in-process profiler (`--stages` forces eager to time stages).

## Quality check
`bench/compare_audio.py --a old --b new`: loudness-normalised log-mel distance, RMVPE pitch error (cents), V/UV agreement, loudness. Compare against a new-vs-new run to know the noise floor (RVC is stochastic).

## Tests
`pytest` (CPU-only in CI): crop exactness, GPU index exactness, pitch helpers vs upstream formulas, RMVPE decode vs upstream loop, HubertLite mapping (+ ONNX parity when `server/pretrain` exists). CI: `.github/workflows/ci.yml`, ruff on paths in `.github/ruff-paths.txt`.
