import numpy as np
from chainer.cuda import cupy as cp

def set_gpu(gpu_index):
    global xp
    xp = np
    if(gpu_index >= 0):
        xp = cp
