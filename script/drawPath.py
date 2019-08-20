#!/usr/bin/env python
import cv2
import pandas as pd
import numpy as np
import data
import xp_settings as setting
setting.set_gpu(-1)

def mouse_callback(e,x,y,flags,param):
    global drawing, m_x, m_y, m_e, m_flags
    if flags == cv2.EVENT_FLAG_LBUTTON:
        m_x = x
        m_y = y
        m_e = e
        m_flags = flags
        drawing = True
    else:
        drawing = False

def getRadian(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    rad = np.arctan2(y2-y1, x2-x1)
    return rad

def main():
    scale = 100.0
    img_w = 500
    img_h = 500
    img = np.ones((img_w,img_h,3),np.uint8)
    img = img * 255
    img_drawn = img.copy()
    cv2.namedWindow('drawPath')
    cv2.setMouseCallback('drawPath',mouse_callback)
    global drawing, m_x, m_y, m_e, m_flags
    drawing = False
    pathData = np.empty((0,2),np.float32)
    while(True):
        cv2.imshow('drawPath',img_drawn)
        k = cv2.waitKey(1) & 0xFF
        if k==ord('q'):
            break;
        if drawing:
            cv2.circle(img_drawn,(m_x,m_y),3,(0,0,0),-1)
            d = np.array((m_x/scale,m_y/scale),np.float32)
            print(d)
            pathData = np.vstack((pathData,d))
            drawing = False
    cv2.destroyAllWindows()
    pathData = pathData - pathData[0]
    rad = data.get_radian(pathData)
    print(pathData.shape)
    print(rad.shape)
    pathData = np.hstack((pathData,rad))
    data.write_path_csv(pathData,'path_draw.csv')
if __name__=='__main__':
    main()
