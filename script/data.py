import pandas as pd
import xp_settings as settings

def set_params(params):
    global DATA_SIZE
    global NUM_WAYPOINTS
    global NUM_STEP
    global HZ
    global MAX_TRANS_VEL
    global MAX_ROTATE_VEL
    global M_PER_STEP
    global RAD_PER_STEP
    DATA_SIZE, NUM_WAYPOINTS, NUM_STEP, HZ, MAX_TRANS_VEL, MAX_ROTATE_VEL, M_PER_STEP, RAD_PER_STEP = params
    print('DATA_SIZE     :',DATA_SIZE     )
    print('NUM_WAYPOINTS :',NUM_WAYPOINTS )
    print('NUM_STEP      :',NUM_STEP      )
    print('HZ            :',HZ            )
    print('MAX_TRANS_VEL :',MAX_TRANS_VEL )
    print('MAX_ROTATE_VEL:',MAX_ROTATE_VEL)
    print('M_PER_STEP    :',M_PER_STEP    )
    print('RAD_PER_STEP  :',RAD_PER_STEP  )

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

def make_arc_path_2(l,alpha,s=1):
    if alpha==0:
        X = settings.xp.linspace(0, l, 1000)
        Y = settings.xp.zeros(len(X))
        rad = settings.xp.zeros(len(X))
        path = settings.xp.stack((X,Y,rad),axis=1)
    else:
        r = s/alpha # curvature
        rad = settings.xp.linspace(-settings.xp.pi/2, -settings.xp.pi/2+l/r, 1000)
        X = r * settings.xp.cos(rad)
        Y = r * settings.xp.sin(rad) + r
        path = settings.xp.stack((X,Y,rad),axis=1)
    return path

def rotate_path(path, rad):
    X = path[:,1]*settings.xp.sin(rad) + path[:,0]*settings.xp.cos(rad)
    Y = path[:,1]*settings.xp.cos(rad) - path[:,0]*settings.xp.sin(rad)
    R = path[:,2] + rad
    ret = settings.xp.stack((X,Y,R),axis=1)
    return ret

def translate_path(path, margin_x, margin_y):
    X = path[:,0] + margin_x
    Y = path[:,1] + margin_y
    R = path[:,2]
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
    #p1 = settings.xp.array(p1,dtype=settings.xp.float32)
    #p2 = settings.xp.array(p2,dtype=settings.xp.float32)
    D = settings.xp.sqrt(settings.xp.sum((p2-p1)**2,axis=ax))
    return D

def old_get_evenly_spaced_points(data,space):
    ret = settings.xp.empty((0,3))
    idx_list = [];
    p1 = (0.0, 0.0, 0.0);
    for i in range(1,len(data)):
        p2 = data[i]
        D = calc_distance(p1[0:2],p2[0:2]) 
        if(D >= space):
            ret = settings.xp.vstack((ret,p2))
            idx_list.append(i)
            p1 = p2
    return ret, idx_list

def get_evenly_spaced_points(data,space):
    ret = settings.xp.empty((0,3))
    idx_list = [];
    p1 = (0.0, 0.0, 0.0);
    for i in range(1,len(data)):
        p2 = data[i]
        D = calc_distance(p1[0:2],p2[0:2]) 
        if(D >= space):
            ret = settings.xp.vstack((ret,data[i-1]))
            idx_list.append(i-1)
            p1 = data[i-1]
    return ret, idx_list

def get_random_spaced_points(data, max_v):
    ret = settings.xp.empty((0,3))
    idx_list = [];
    p1 = (0.0, 0.0, 0.0);
    for i in range(1,len(data)):
        p2 = data[i]
        D = calc_distance(p1[0:2],p2[0:2]) 
        if(D >= (settings.xp.random.rand() * max_v) ):
            ret = settings.xp.vstack((ret,data[i-1]))
            idx_list.append(i-1)
            p1 = data[i-1]
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
    dist = calc_distance(point[0:2],pathData[:,0:2],ax=1)
    idx = settings.xp.argmin(dist)
    return idx

def get_next_path_idx(idx,idx_list):
    next_idx = 0
    for i in idx_list:
        if idx < i:
            break
        next_idx = next_idx + 1
    return next_idx
    
def get_waypoints(close_idx, data,num_step,space):
    ret = settings.xp.zeros((num_step,3),dtype=settings.xp.float32)
    idx = close_idx
    p1 = data[idx];
    for n in range(num_step):
        D = calc_distance(p1[0:2],data[idx:,0:2],ax=1)
        D = settings.xp.abs(D - space)
        idx = idx + settings.xp.argmin(D)
        if(settings.xp.argmin(D) == 0 ):
            ret = []
            break
        ret[n,:] = data[idx]
        p1 = data[idx]
    return ret

def generate_arc_path(num_step,rad_per_step,m_per_step):
    digit = 10 ** 5
    rad_per_step = float(int(rad_per_step * digit) / digit)
    if(rad_per_step == 0):
        X = settings.xp.linspace(.0, m_per_step*num_step, num_step,endpoint=False) + m_per_step
        Y = settings.xp.zeros(len(X))
        TH = settings.xp.zeros(len(X))
        path = settings.xp.stack((X,Y,TH),axis=1)
    else:
        r = m_per_step / rad_per_step
        start_th = -settings.xp.pi/2
        TH = settings.xp.linspace(start_th, start_th+(rad_per_step*num_step), num_step, endpoint=False) + rad_per_step
        X = r * settings.xp.cos(TH)
        Y = r * settings.xp.sin(TH) + r
        path = settings.xp.stack((X,Y,TH),axis=1)
    return path

