import torch

# Get the PyTorch version
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# Check if CUDA is available (indicating GPU support)
is_cuda_available = torch.cuda.is_available()
print(f"CUDA available: {is_cuda_available}")

# Determine the type of PyTorch version
if is_cuda_available:
    print("This is the GPU version of PyTorch.")
else:
    print("This is the CPU version of PyTorch.")
