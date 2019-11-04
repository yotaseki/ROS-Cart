import cv2
import pandas as pd
import numpy as np
from scipy import interpolate
import datetime

def main():
    p_curve = np.empty((0,2),dtype=np.uint8)
    p_clicked = np.empty((0,2),dtype=np.uint8)
    scale = 0.005
    img_w = 1000 # m
    img_h = 1000
    img = np.ones((img_w,img_h,3),np.uint8)
    img = img * 255
    img_drawn = img.copy()
    cv2.namedWindow('drawPath')
    cv2.setMouseCallback('drawPath',mouse_callback)
    global mx,my,left_down
    left_down = False
    while(True):
        cv2.imshow('drawPath',img_drawn)
        k = cv2.waitKey(1) & 0xFF
        if left_down:
            #p = np.array([float(mx)/img_w,float(my)/img_h])
            p = np.array([mx,my])
            p_clicked = np.vstack((p_clicked,p))
            if len(p_clicked) >=3:
                #p_curve = np.vstack((p_curve, spline_splprep(p_clicked[:,0], p_clicked[:,1], deg=2)))
                p_curve = spline_splprep(p_clicked[:,0], p_clicked[:,1], deg=2, num_point=1000*len(p_clicked))
                img_drawn = img.copy()
                for px,py in p_curve:
                    px = int(px)
                    py = int(py)
                    cv2.circle(img_drawn,(px,py),3,(255,0,0),-1)
                #p_clicked = np.empty((0,2),dtype=np.uint8)
                #p_clicked = np.vstack((p_clicked,p))
            else:
                for px,py in p_clicked:
                    cv2.circle(img_drawn,(px,py),3,(0,0,0),-1)
            left_down = False
        if k==ord('c'):
            img_drawn = img.copy()
            p_clicked = np.empty((0,2),dtype=np.uint8)
            p_curve = np.empty((0,2),dtype=np.uint8)
        if k==ord('q'):
            break;
    cv2.destroyAllWindows()

    p_curve = p_curve - p_curve[0]
    p_curve = p_curve * scale
    rad = getRadian(np.transpose(p_curve[:len(p_curve)-1]),np.transpose(p_curve[1:]))
    rad = np.insert(rad,0,0.0)
    p_curve = np.hstack((p_curve,np.transpose([rad])))
    print(rad.shape)
    print(p_curve.shape)
    p_curve = np.round(p_curve,decimals=8)
    df = pd.DataFrame(p_curve)
    df.to_csv(get_filename('curve_spline','.csv'),float_format= '%.8f',header=None,index=False)

def get_filename(head,ext):
    d = '{0:%Y%m%d%H%M}'.format(datetime.datetime.now())
    fn = head + d + ext
    return fn

def mouse_callback(e,x,y,flags,param):
    global mx,my,left_down
    mx = x
    my = y
    if e == cv2.EVENT_LBUTTONDOWN:
        left_down = True

def spline_splprep(x,y,num_point=1000,deg=3):
    tck,u = interpolate.splprep([x,y],k=deg,s=0) 
    u = np.linspace(0,1,num=num_point,endpoint=True) 
    spline = interpolate.splev(u,tck)
    spline = np.transpose(spline)
    return spline

def getRadian(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    rad = np.arctan2(y2-y1, x2-x1)
    return rad

if __name__=='__main__':
    main()
