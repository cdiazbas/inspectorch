import platform
import socket
import torch

def device_info():
    # Small utility to print device information (if cuda is available and how many GPUs and names):
    if torch.cuda.is_available():
        print(f"Machine name: {socket.gethostname()}")
        print(f"Operating system: {platform.system()} {platform.release()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")
