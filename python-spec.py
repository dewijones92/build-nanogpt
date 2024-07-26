import torch
import os

print('CUDA:', torch.version.cuda)

cudnn = torch.backends.cudnn.version()
cudnn_major = cudnn // 1000
cudnn = cudnn % 1000
cudnn_minor = cudnn // 100
cudnn_patch = cudnn % 100
print('cuDNN:', '.'.join([str(cudnn_major), str(cudnn_minor), str(cudnn_patch)]))

# Attempt to find cuDNN frontend path
possible_paths = [
    '/usr/local/cuda/include/cudnn_frontend.h',
    '/usr/include/cudnn_frontend.h',
    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\include\\cudnn_frontend.h'
]

cudnn_frontend_path = None
for path in possible_paths:
    if os.path.exists(path):
        cudnn_frontend_path = path
        break

if cudnn_frontend_path:
    print(f"cuDNN frontend path: {cudnn_frontend_path}")
else:
    print("cuDNN frontend path not found in common locations.")
    print("You may need to manually specify the path if it's in a different location.")