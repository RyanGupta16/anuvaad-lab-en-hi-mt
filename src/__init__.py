"""English→Hindi MT benchmark package."""
import os

# Keep the Apple-silicon (MPS) caching allocator inside physical memory: without these caps
# it grows past unified RAM on variable-shape batches and the machine starts swapping.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
