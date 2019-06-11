#!/usr/bin/env python
import cv2
import pandas as pd
import numpy as np
from data import PathData

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
    pathData = np.empty((0,3),np.float32)
    pos_d = np.array((0,0),np.uint8)
    while(True):
        cv2.imshow('drawPath',img_drawn)
        k = cv2.waitKey(1) & 0xFF
        if k==27:
            break;
        if drawing:
            cv2.circle(img_drawn,(m_x,m_y),3,(0,0,0),-1)
            pos = np.array((m_x, m_y),np.uint8)
            rad = 0.0
            if len(pathData) > 0:
                rad = getRadian(pos_d,pos)
                print(data)
            data = np.array((pos[0]/scale,pos[1]/scale,rad),np.float32)
            pathData = np.vstack((pathData,data))
            drawing = False
            pos_d = pos.copy()
    cv2.destroyAllWindows()
    pathData = pathData - pathData[0]
    pd = PathData()
    pd.write_path_csv(pathData,'path_full.csv')
if __name__=='__main__':
    main()
