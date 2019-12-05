import chainer
from chainer import Link, Chain, ChainList, Variable, optimizers, iterators
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer import serializers

import xp_settings as settings
import options
from model import calc_oplus
import data

def error_squares(z_oplus,z_true):
    w = settings.xp.array([1., 1., 0.],dtype=settings.xp.float32)
    t = z_true * w
    p = z_oplus * w
    error = F.sqrt(F.sum((t-p)*(t-p)))
    return error

def error_diff_v(y0, y1):
    w = settings.xp.array([0., 0., 1.],dtype=settings.xp.float32)
    v0 = y0 * w
    v1 = y1 * w
    error = F.absolute(F.sum(v1 - v0))
    return error

def error_diff_w(y0, y1):
    w = settings.xp.array([0., 0., 1.],dtype=settings.xp.float32)
    v0 = y0 * w
    v1 = y1 * w
    error = F.absolute(F.sum(v1 - v0))
    return error

def loss_function(y, y_t):
    alpha = 1.0
    beta = 10.0
    dmesg = ''
    # distance - stage cost
    cost_st = error_squares(y[0],y_t[0])
    for i in range(1,options.DATA_NUM_WAYPOINTS-1):
        cost_st = cost_st + error_squares(y[i],y_t[i])
    # distance - terminal cost
    cost_term = error_squares(y[-1],y_t[-1])
    loss = alpha*cost_st + beta*cost_term
    return loss

# training
def train(model,opt,X,epoch=10,clipping=0):
    AvgLoss = []
    for ep in range(epoch):
        random.shuffle(X)
        L = .0
        for itr in range(len(X)):
            prev_u = settings.xp.zeros((options.DATA_NUM_PREVIOUS_U,2),dtype=settings.xp.float32)
            x = settings.xp.vstack((X[itr][:,0:2],prev_u))
            # forward
            x = settings.xp.ravel(x_data)
            x = Variable(settings.xp.array([x],dtype=settings.xp.float32))
            uv, uw = model(x)
            # clip
            if clipping:
                v_limit = options.DATA_MAX_V_STEP
                w_limit = options.DATA_MAX_W_STEP
                v = F.clip(uv,0.0,v_limit)
                w = F.clip(uw,-w_limit,w_limit)
            else:
                v = uv
                w = uw
            # zero-padding
            pad = Variable(settings.xp.zeros((1,options.DATA_NUM_STEP),dtype=settings.xp.float32))
            u = F.stack((v,pad,w),axis=2)
            z = calc_oplus(u)
            z_t = x_data[:-options.DATA_NUM_PREVIOUS_U]
            # loss
            loss = loss_function(z, z_t)
            # update
            model.cleargrads()
            loss.backward()
            opt.update()
            #y_grad = chainer.grad( (e,), (y,) )[0]
            #print(y_grad)
            #x_grad = chainer.grad( (e,), (x,) )[0]
            #print(x_grad)
            L = L + loss
        print('Epoch:',ep,', Average loss:',L / len(X))
        AvgLoss.append(L / len(X))
    return model, AvgLoss

def make_dataset():
    X_train = []
    for i in range(options.DATA_SIZE):
        rand_rad = settings.xp.random.rand()*(2*options.DATA_W_STEP)-options.DATA_W_STEP # -36 ~ 36 m/step
        d = data.generate_arc_path(options.DATA_NUM_STEP,rand_rad,options.DATA_V_STEP)
        d = data.rotate_path(d,rand_rad*0.5)
        if options.DATA_RANGE_TRANSLATE != 0:
            rand_trans_x = settings.xp.random.rand() * options.DATA_RANGE_TRANSLATE
            d = data.translate_path(d,rand_trans_x,0.0)
        if options.DATA_RANGE_ROTATE != 0:
            rand_rotate = settings.xp.random.rand()*(options.DATA_RANGE_ROTATE*2)-options.DATA_RANGE_ROTATE
            d = data.rotate_path(d,rand_rotate)
        X_train.append(d)
    return X_train

