'''
other useful commands:
nvidia-smi
nvcc --version
'''
import torch

print("torch version")
print(torch.__version__)

print ("torch.cuda.device(0)")
print(torch.cuda.device(0))

print("torch.cuda.current_device()")
print(torch.cuda.current_device())

print("torch.cuda.device_count()")
print(torch.cuda.device_count())

print("torch.cuda.is_available()")
print(torch.cuda.is_available())
