from chainer import Link,Chain
from chainer import Function,Variable
from chainer import initializers, serializers
import chainer.functions as F
import chainer.links as L
from chainer.cuda import cupy as cp
import numpy as np

class Generator(Chain):
    def __init__(self, point_num, num_step):
        self.input_dim = point_num * 2
        self.num_step = num_step
        self.output_dim = num_step * 2
        self.l1_dim = 15
        self.l2_dim = 8
        initializer = initializers.HeNormal()
        super(Generator, self).__init__(
            l1=L.Linear(self.input_dim, self.l1_dim, initialW=initializer),
            l2=L.Linear(self.l1_dim, self.l2_dim, initialW=initializer),
            l3=L.Linear(self.l2_dim, self.output_dim, initialW=initializer),
        )
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        h4 = F.tanh(h3)
        o = F.reshape(h4,(len(x),self.num_step,2))
        return o

class Oplus(Function):
    def forward_cpu(self, inputs):
        t1, t2 = inputs
        #self.retain_inputs((0, 1))
        cos1 = np.cos(t1[:,2])
        sin1 = np.sin(t1[:,2])
        x = cos1 * t2[:,0] - sin1 * t2[:,1] + t1[:,0]
        y = sin1 * t2[:,0] + cos1 * t2[:,1] + t1[:,1]
        t = ( t1[:,2] + t2[:,2] + np.pi) % (2 * np.pi ) - np.pi
        ret = np.array([x,y,t], dtype=t1.dtype)
        ret = ret.transpose(),
        #print('ret:' + str(ret))
        return ret
    
    def backward_cpu(self, inputs, grad_outputs):
        #t1, t2 = self.retained_inputs()
        t1, t2 = inputs
        in_len = len(inputs)
        gw, = grad_outputs
        #print('go')
        #print(grad_outputs)
        cos1 = np.cos(t1[:,2])
        sin1 = np.sin(t1[:,2])
        dx1 = np.zeros((in_len,3,3), dtype=gw.dtype)
        dx1[:,0,0] = 1
        dx1[:,1,1] = 1
        dx1[:,2,2] = 1
        dx1[:,0,2] = -sin1 * t2[:,0] - cos1 *t2[:,1]
        dx1[:,1,2] = cos1 * t2[:,0] - sin1 *t2[:,1]
        #print('dx1')
        #print(dx1)
        gw = gw.reshape(in_len,1,3)
        #print(gw.shape)
        #print(dx1.shape)
        d1 = np.squeeze(np.matmul(gw, dx1))
        dx2 = np.zeros((in_len,3,3), dtype=gw.dtype)
        dx2[:,0,0] = cos1
        dx2[:,0,1] = -sin1
        dx2[:,1,0] = sin1
        dx2[:,1,1] = cos1
        dx2[:,2,2] = 1
        #print(gw.shape)
        #print(dx2.shape)
        d2 = np.squeeze(np.matmul(gw, dx2))
        return d1, d2

    def forward_gpu(self, inputs):
        t1, t2 = inputs
        #self.retain_inputs((0, 1))
        cos1 = cp.cos(t1[:,2])
        sin1 = cp.sin(t1[:,2])
        x = cos1 * t2[:,0] - sin1 * t2[:,1] + t1[:,0]
        y = sin1 * t2[:,0] + cos1 * t2[:,1] + t1[:,1]
        t = ( t1[:,2] + t2[:,2] + cp.pi) % (2 * cp.pi ) - cp.pi
        ret = cp.array([x,y,t], dtype=t1.dtype)
        ret = ret.transpose(),
        #print('ret:' + str(ret))
        return ret
    
    def backward_gpu(self, inputs, grad_outputs):
        #t1, t2 = self.retained_inputs()
        t1, t2 = inputs
        in_len = len(inputs)
        gw, = grad_outputs
        #print('go')
        #print(grad_outputs)
        cos1 = cp.cos(t1[:,2])
        sin1 = cp.sin(t1[:,2])
        dx1 = cp.zeros((in_len,3,3), dtype=gw.dtype)
        dx1[:,0,0] = 1.
        dx1[:,1,1] = 1.
        dx1[:,2,2] = 1.
        dx1[:,0,2] = -sin1 * t2[:,0] - cos1 *t2[:,1]
        dx1[:,1,2] = cos1 * t2[:,0] - sin1 *t2[:,1]
        #print('dx1')
        #print(dx1)
        gw = gw.reshape(in_len,1,3)
        #print(gw.shape)
        #print(dx1.shape)
        d1 = cp.squeeze(cp.matmul(gw, dx1))
        dx2 = cp.zeros((in_len,3,3), dtype=gw.dtype)
        dx2[:,0,0] = cos1
        dx2[:,0,1] = -sin1
        dx2[:,1,0] = sin1
        dx2[:,1,1] = cos1
        dx2[:,2,2] = 1.
        d2 = cp.squeeze(cp.matmul(gw, dx2))
        return d1, d2

# def oplus(x, y):
#     return Oplus()(x,y)
