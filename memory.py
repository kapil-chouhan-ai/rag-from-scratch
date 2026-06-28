"""
Memory-footprint tracking, used by eval/benchmark.py to produce the
"memory footprint" figure across retrieval configurations.

This file previously just printed CUDA memory stats once at import time
while the README described it as a "persistence layer for vector store
and document store" - that persistence logic actually lives in
rag/ingestion.py's save()/load(). Rather than leave the mismatch, this
is now an actual, reusable utility: process RSS via psutil (works on
Windows/Linux/Mac) plus GPU allocation via torch when CUDA is available.
"""

import psutil

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    _CUDA_AVAILABLE = False


class MemoryTracker:

    def __init__(self):
        self._process = psutil.Process()

    def snapshot(self):
        stats = {"rss_mb": round(self._process.memory_info().rss / 1024**2, 2)}

        if _CUDA_AVAILABLE:
            stats["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 2)
            stats["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 2)

        return stats

    def delta(self, before, after):
        return {key: round(after[key] - before.get(key, 0), 2) for key in after}