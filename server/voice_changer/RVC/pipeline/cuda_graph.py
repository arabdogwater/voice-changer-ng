"""Replay the whole per-chunk GPU workload as one CUDA graph.

A streaming chunk launches ~3000 small kernels (RMVPE, HuBERT, synthesizer). On Windows/WDDM the
CPU cost of launching them (~70 ms) dwarfed the GPU time (~20 ms). For a fixed configuration the
shapes never change, so the work is captured once and replayed with a single launch.
"""

from collections import OrderedDict
from typing import Callable, Hashable, Sequence

import torch
from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()


class GraphCache:
    def __init__(self, max_graphs: int = 4, warmup: int = 2):
        self.max_graphs = max_graphs
        self.warmup = warmup
        self.graphs: OrderedDict[Hashable, tuple[list[torch.Tensor], tuple[torch.Tensor, ...], torch.cuda.CUDAGraph]] = OrderedDict()
        self.disabled = False

    def clear(self):
        self.graphs.clear()

    def run(self, key: Hashable, fn: Callable[..., tuple[torch.Tensor, ...]], inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(fn, inputs)
            self.graphs[key] = entry
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        static_in, static_out, graph = entry
        for dst, src in zip(static_in, inputs):
            dst.copy_(src, non_blocking=True)
        graph.replay()
        # outputs live in the graph's private pool and are overwritten by the next replay
        return tuple(o.clone() for o in static_out)

    def _capture(self, fn, inputs):
        static_in = [x.detach().clone() for x in inputs]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up: lazy buffers, cuDNN/cuBLAS handles, autotuning
            for _ in range(self.warmup):
                fn(*static_in)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(*static_in)
        out = out if isinstance(out, tuple) else (out,)
        logger.info(f"[Voice Changer] captured CUDA graph ({len(self.graphs) + 1} cached)")
        return static_in, out, graph
