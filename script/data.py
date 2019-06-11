import pandas as pd
import numpy
from chainer.cuda import cupy

class PathData:
    def __init__(self, gpu_idx=-1):
        if(gpu_idx >= 0):
            self.xp = cupy
        else:
            self.xp = numpy

    def make_function_path(self, f,x):
        X = x
        Y = f(X)
        path = self.xp.stack((X,Y),axis=1)
        rad = self.get_radian(path)
        path = self.xp.hstack(path,rad)
        return path

    def make_arc_path(self, l,r):
        rad = self.xp.linspace(-self.xp.pi/2, -self.xp.pi/2+l/r)
        X = r * self.xp.cos(rad)
        Y = r * self.xp.sin(rad) + r
        path = self.xp.stack((X,Y,rad),axis=1)
        return path

    def make_arc_path_2(self, l,alpha,s=1):
        if alpha==0:
            X = self.xp.linspace(0, l)
            Y = self.xp.zeros(len(X))
            rad = self.xp.zeros(len(X))
            path = self.xp.stack((X,Y,rad),axis=1)
        else:
            r = s/alpha # curvature
            rad = self.xp.linspace(-self.xp.pi/2, -self.xp.pi/2+l/r)
            X = r * self.xp.cos(rad)
            Y = r * self.xp.sin(rad) + r
            path = self.xp.stack((X,Y,rad),axis=1)
        return path

    def rotate_path(self, path, deg):
        rad = self.xp.deg2rad(deg)
        X = path[:,1]*self.xp.cos(rad) - path[:,0]*self.xp.sin(rad)
        Y = path[:,1]*self.xp.sin(rad) + path[:,0]*self.xp.cos(rad)
        R = path[:,2] + rad
        ret = self.xp.stack((X,Y,R),axis=1)
        return ret

    def read_path_csv(self, filename):
        df = pd.read_csv(filename, header=None)
        path = self.xp.array(df.values,dtype=self.xp.float32)
        return path
    
    def write_path_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename,header=None,index=False)

    def calc_distance(self,p1,p2,ax=0):
        D = self.xp.sqrt(self.xp.sum((p1-p2)**2),axis=ax)
        return D

    def get_evenly_spaced_points(self,data,space):
        ret = self.xp.empty((0,3))
        idx_list = [];
        p1 = (0.0, 0.0, 0.0);
        for i in range(len(data)):
            D = self.calc_distance(p1[i,0:1],p2[i,0:1]) 
            if(D >= space):
                ret = self.xp.vstack((ret,p2))
                idx_list.append(i)
                p1 = p2
        return ret, idx_list
    
    def get_n_point_from_path(self, n,path,margin=5):
        ret = self.xp.empty((0,3))
        idx = 0
        for i in range(n):
            idx = idx + self.xp.random.randint(margin,len(path)/3)
        if(idx > len(path)):
            idx = len(path)
            ret = self.xp.vstack((ret,path[idx]))
        return ret

    def get_n_point_from_path_2(self, n,path):
        ret = self.xp.empty((0,3))
        idx = [(len(path)-1)/3, (len(path)-1)*2/3, len(path)-1]
        for i in idx:
            i = int(i)
            ret = self.xp.vstack((ret,path[i]))
        return ret

    def get_radian(self,path):
        pos = path[0:len(path)-1]
        pos_d = path[1:len(path)]
        rad = self.calc_radian(pos,pos_d)
        rad = self.xp.append(rad,0.0)
        rad = self.xp.expand_dims(rad,axis=1)
        return rad

    def get_nearly_point_idx(pathData,point):
        dist = calc_distance(PathData[:,0:2],point,ax=1)
        idx = self.xp.argmin(dist)

    def calc_radian(self,p1,p2):
        rad = self.xp.arctan2(p2[:,1]-p1[:,1], p2[:,0]-p1[:,0])
        return rad
    
