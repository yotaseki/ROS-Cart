import pandas as pd
import xp_settings as settings

def calc_radian(p1,p2):
    rad = settings.xp.arctan2(p2[:,1]-p1[:,1], p2[:,0]-p1[:,0])
    return rad

def get_radian(path):
    pos = path[0:len(path)-1]
    pos_d = path[1:len(path)]
    rad = calc_radian(pos,pos_d)
    rad = settings.xp.append(rad,0.0)
    rad = settings.xp.expand_dims(rad,axis=1)
    return rad

def make_function_path(f,x):
    X = x
    Y = f(X)
    path = settings.xp.stack((X,Y),axis=1)
    rad = get_radian(path)
    path = settings.xp.hstack(path,rad)
    return path

def make_arc_path(l,r):
    rad = settings.xp.linspace(-settings.xp.pi/2, -settings.xp.pi/2+l/r)
    X = r * settings.xp.cos(rad)
    Y = r * settings.xp.sin(rad) + r
    path = settings.xp.stack((X,Y,rad),axis=1)
    return path

def make_arc_path_2(l,alpha,s=0.1):
    if alpha==0:
        X = settings.xp.linspace(0, l)
        Y = settings.xp.zeros(len(X))
        rad = settings.xp.zeros(len(X))
        path = settings.xp.stack((X,Y,rad),axis=1)
    else:
        r = s/alpha # curvature
        rad = settings.xp.linspace(-settings.xp.pi/2, -settings.xp.pi/2+l/r)
        X = r * settings.xp.cos(rad)
        Y = r * settings.xp.sin(rad) + r
        path = settings.xp.stack((X,Y,rad),axis=1)
    return path

def rotate_path(path, deg):
    rad = settings.xp.deg2rad(deg)
    X = path[:,1]*settings.xp.cos(rad) - path[:,0]*settings.xp.sin(rad)
    Y = path[:,1]*settings.xp.sin(rad) + path[:,0]*settings.xp.cos(rad)
    R = path[:,2] + rad
    ret = settings.xp.stack((X,Y,R),axis=1)
    return ret

def read_path_csv(filename):
    df = pd.read_csv(filename, header=None)
    path = settings.xp.array(df.values,dtype=settings.xp.float32)
    return path

def write_path_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename,header=None,index=False)

def calc_distance(p1,p2,ax=0):
    p1 = settings.xp.array(p1)
    p2 = settings.xp.array(p2)
    D = settings.xp.sqrt(settings.xp.sum((p1-p2)**2,axis=ax))
    return D

def get_evenly_spaced_points(data,space):
    ret = settings.xp.empty((0,3))
    idx_list = [];
    p1 = (0.0, 0.0, 0.0);
    for i in range(len(data)):
        p2 = data[i]
        D = calc_distance(p1[0:2],p2[0:2]) 
        if(D >= space):
            ret = settings.xp.vstack((ret,p2))
            idx_list.append(i)
            p1 = p2
    return ret, idx_list

def get_n_point_from_path(n,path,margin=5):
    ret = settings.xp.empty((0,3))
    idx = 0
    for i in range(n):
        idx = idx + settings.xp.random.randint(margin,len(path)/3)
    if(idx > len(path)):
        idx = len(path)
        ret = settings.xp.vstack((ret,path[idx]))
    return ret

def get_n_point_from_path_2(n,path):
    ret = settings.xp.empty((0,3))
    idx = [(len(path)-1)/3, (len(path)-1)*2/3, len(path)-1]
    for i in idx:
        i = int(i)
        ret = settings.xp.vstack((ret,path[i]))
    return ret

def get_nearly_point_idx(pathData,point):
    dist = calc_distance(pathData[:,0:2],point[0:2],ax=1)
    idx = settings.xp.argmin(dist)
    return idx

def get_next_path_idx(idx,idx_list):
    next_idx = 0
    for i in idx_list:
        if idx < i:
            break
        next_idx = next_idx + 1
    return next_idx
    

