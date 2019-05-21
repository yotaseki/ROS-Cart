import numpy as np
from chainer.cuda import cupy as cp

class PathData:
    def __init__(self, gpu_idx=-1):
        if(gpu_idx >= 0):
            self.xp = cp
        else:
            self.xp = np

    # 関数による経路生成
    def make_function_path(self, f,x):
        X = x
        Y = f(X)
        path = self.xp.stack((Y,X),axis=1)
        return path

    # 長さと半径から曲率一定の経路を生成
    def make_arc_path(self, l,r):
        theta = self.xp.linspace(-self.xp.pi/2, -self.xp.pi/2+l/r)
        X = r * self.xp.cos(theta)
        Y = r * self.xp.sin(theta) + r
        path = self.xp.stack((Y,X),axis=1)
        return path

    def make_arc_path_2(self, l,alpha,s=1):
        if alpha==0:
            X = self.xp.linspace(0, l)
            Y = self.xp.zeros(len(X))
            path = self.xp.stack((Y,X),axis=1)
        else:
            r = s/alpha # 曲率
            theta = self.xp.linspace(-self.xp.pi/2, -self.xp.pi/2+l/r)
            X = r * self.xp.cos(theta)
            Y = r * self.xp.sin(theta) + r
            path = self.xp.stack((Y,X),axis=1)
        return path

    # 経路を回転
    def rotate_path(self, path, rad):
        theta = rad
        X = path[:,1]*self.xp.cos(theta) - path[:,0]*self.xp.sin(theta)
        Y = path[:,1]*self.xp.sin(theta) + path[:,0]*self.xp.cos(theta)
        ret = self.xp.stack((Y,X),axis=1)
        return ret

    # CSVから経路の読み取り
    def read_path_csv(self, filename):
        df = pd.read_csv(filename, header=None)
        path = self.xp.array(df.values,dtype=self.xp.float32)
        return path

    # 経路からN点抜き出す
    def get_n_point_from_path(self, n,path,margin=5):
        path_data = self.xp.empty((1,2))
        idx = 0
        for i in range(n):
            idx = idx + self.xp.random.randint(margin,len(path)/3)
        if(idx > len(path)):
            idx = len(path)
            path_data = self.xp.vstack((path_data,path[idx]))
        return path_data[1:len(path_data)]

    def get_n_point_from_path_2(self, n,path):
        path_data = self.xp.empty((1,2))
        idx = [(len(path)-1)/3, (len(path)-1)*2/3, len(path)-1]
        for i in idx:
            i = int(i)
            path_data = self.xp.vstack((path_data,path[i]))
        return path_data[1:len(path_data)]
