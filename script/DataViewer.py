import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import calc_oplus
import coordinate
import xp_settings as settings
settings.set_gpu(-1)
xp = settings.xp

if __name__=='__main__':
    d = sys.argv[1]
    logx = d + '/log_x.csv'
    logpos = d + '/log_pos.csv'
    logv = d + '/log_v.csv'
    logw = d + '/log_w.csv'
    x = pd.read_csv(logx, header=None).values
    x = np.reshape(x,(len(x),10,2))
    pos = pd.read_csv(logpos, header=None).values
    pos = np.reshape(pos,(len(x),10,3))
    v = pd.read_csv(logv,header=None).values
    w = pd.read_csv(logw,header=None).values
    pad = np.zeros((len(v),10),dtype=np.float32)
    y_pad = np.stack([v,pad,w],axis=2)
    print('x:',x.shape)
    print('pos:',pos.shape)
    print('y:',y_pad.shape)
    for i in range(len(x)):
        z = np.array(calc_oplus(y_pad[i]))
        p = coordinate.globalpos_to_localpos(pos[i], pos[i,0])
        plt.plot(x[i,:,0],x[i,:,1])
        plt.plot(p[:,0],p[:,1])
        plt.plot(z[:,0],z[:,1])
        plt.legend(['x','pos','z'])
        plt.show()

