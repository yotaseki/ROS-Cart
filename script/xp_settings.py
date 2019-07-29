import numpy as np
from chainer.cuda import cupy as cp

def set_gpu(gpu_idx):
    global xp, gpu_index
    gpu_index = gpu_idx
    xp = np
    if(gpu_index >= 0):
        xp = cp
