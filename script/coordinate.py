import xp_settings as settings

def translate(data,trans_x,trans_y):
    X = data[:,0] - trans_x
    Y = data[:,1] - trans_y
    TH = data[:,2]
    ret = settings.xp.stack((X,Y,TH),axis=1)
    return ret

def rotate(data,rad):
    X = data[:,1]*settings.xp.cos(rad) - data[:,0]*settings.xp.sin(rad)
    Y = data[:,1]*settings.xp.sin(rad) + data[:,0]*settings.xp.cos(rad)
    TH = data[:,2] + rad
    ret = settings.xp.stack((X,Y,TH),axis=1)
    return ret

def globalpos_to_localpos(data,selfpos):
    x, y, th = selfpos
    ret = rotate(data,-th)
    ret = translate(ret,x,y)
    return ret

