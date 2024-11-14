import torch

# Check if CUDA (GPU support) is available
gpu_available = torch.cuda.is_available()
print(f"CUDA available: {gpu_available}")

# If CUDA is available, get the name of the GPU
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
else:
    print("No GPU detected.")
