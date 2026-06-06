import torch
# Get currently allocated memory by active tensors (in MB)
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Get total memory being held by the PyTorch cache (in MB)
print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")