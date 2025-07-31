import torch

def print_gpu_memory(tag=""):
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**2  # en MB
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] GPU memory - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
