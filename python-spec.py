import torch
print('CUDA:',torch.version.cuda)

cudnn = torch.backends.cudnn.version()
cudnn_major = cudnn // 1000
cudnn = cudnn % 1000
cudnn_minor = cudnn // 100
cudnn_patch = cudnn % 100
print( 'cuDNN:', '.'.join([str(cudnn_major),str(cudnn_minor),str(cudnn_patch)]) )