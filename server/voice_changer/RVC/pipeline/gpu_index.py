import numpy as np
import torch

# score matrix budget per matmul (elements); keeps peak VRAM ~128 MB even for huge indexes
_MAX_SCORE_ELEMS = 32 * 1024 * 1024


class GpuIndex:
    """Exact nearest-neighbour feature retrieval on the GPU.

    Replaces the per-chunk `faiss.search` on CPU (plus the GPU->CPU->GPU copies around it).
    Exact search is also better than the IVF/nprobe=1 approximation most RVC indexes use.
    """

    def __init__(self, vectors: np.ndarray, device: torch.device, half: bool):
        dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
        self.vecs = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)).to(device, dtype)
        self.half_norms = 0.5 * self.vecs.float().pow(2).sum(1)
        self.n = self.vecs.shape[0]

    @torch.inference_mode()
    def search(self, q: torch.Tensor) -> torch.Tensor:
        """q: [T, C] -> nearest stored vectors [T, C] (L2), in q's dtype."""
        q = q.to(self.vecs.dtype)
        step = max(1, _MAX_SCORE_ELEMS // max(1, q.shape[0]))
        best_score, best_ix = None, None
        for s in range(0, self.n, step):
            # argmin ||q - v||^2  ==  argmax q.v - ||v||^2 / 2
            score = (q @ self.vecs[s : s + step].T).float() - self.half_norms[s : s + step]
            sc, ix = score.max(1)
            if best_score is None:
                best_score, best_ix = sc, ix
            else:
                better = sc > best_score
                best_score = torch.where(better, sc, best_score)
                best_ix = torch.where(better, ix + s, best_ix)
        return self.vecs[best_ix]


def load_index_vectors(index_path: str) -> np.ndarray | None:
    import faiss  # only needed to read the .index file once

    index = faiss.read_index(index_path)
    try:
        return index.reconstruct_n(0, index.ntotal)
    except RuntimeError:
        ivf = faiss.extract_index_ivf(index)
        ivf.make_direct_map()
        return index.reconstruct_n(0, index.ntotal)
