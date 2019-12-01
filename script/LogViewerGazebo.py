import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import sys, os, glob
import data
from model import oplus
import xp_settings
import time
xp_settings.set_gpu(-1)
xp = xp_settings.xp

def onClick(e):
    global frame_id
    if e.xdata == None or e.ydata == None:
        return -1
    mx = float(e.xdata)
    my = float(e.ydata)
    idx = data.get_nearly_point_idx(pos_t,[mx,my,0.0])
    frame_id = idx
    replot_fig1(mx,my,pos_t[idx,0],pos_t[idx,1])
    replot_fig2(y_pad[idx],x[idx])
    replot_fig3(pos_t[idx])
    plt.draw()
    print('frame:',idx)
    print('pos_t:',pos_t[idx])
    print('** input **')
    print(x[idx])
    print('** output **')
    print(y_pad[idx])
    print('')
    return 0

def onPressed(e):
    global frame_id
    # print(e.key)
    if e.key == 'p':
        while frame_id<len(pos_t):
            idx = frame_id + 1 
            frame_id = idx
            replot_fig1(pos_t[idx,0],pos_t[idx,1],pos_t[idx,0],pos_t[idx,1])
            replot_fig2(y_pad[idx],x[idx])
            replot_fig3(pos_t[idx])
            plt.draw()
            plt.pause(.001)
            if idx != frame_id:
                break
    if e.key == 'F':
        frame_id = frame_id + 1
        idx = frame_id
        replot_fig1(pos_t[idx,0],pos_t[idx,1],pos_t[idx,0],pos_t[idx,1])
        replot_fig2(y_pad[idx],x[idx])
        replot_fig3(pos_t[idx])
        plt.draw()
        plt.pause(.001)
    if e.key == 'escape':
        sys.exit()

def replot_fig1(mx,my,px,py):
    #ln1_mouse.set_data((mx,px),(my,py))
    #p1_mouse.set_data(mx,my)
    p1_pos.set_data(px,py)

def replot_fig2(y,x):
    p2_waypoint.set_data(x[:-1,0],x[:-1,1])
    z = np.array(calc_oplus(y))
    X = z[0:len(z)-1,0]
    Y = z[0:len(z)-1,1]
    U = z[1:len(z),0] - X
    V = z[1:len(z),1] - Y
    U = np.insert(U,0,X[0])
    V = np.insert(V,0,Y[0])
    X = np.insert(X,0,0.0)
    Y = np.insert(Y,0,0.0)
    p2_arrow.set_offsets(np.stack([X,Y],axis=1))
    p2_arrow.set_UVC(U,V)
    ax3.set_title('frame : '+str(frame_id)+'/'+str(len(pos_t)))

def replot_fig3(pos):
    global radius,delta
    x,y,th = pos
    X,Y,U,V = delta_arrow(x,y,th,delta=delta)
    p3_selfpos.set_offsets([X,Y])
    p3_selfpos.set_UVC(U,V)
    #path_idx = data.get_nearly_point_idx(path,pos)
    #x,y,th = path[path_idx]
    #X,Y,U,V = delta_arrow(x,y,th,delta=delta)
    #p3_truepos.set_offsets([X,Y])
    #p3_truepos.set_UVC(U,V)
    ax3.set_xlim([x-radius,x+radius])
    ax3.set_ylim([y-radius,y+radius])

def calc_oplus(y):
    dst = []
    z = y[0]
    dst.append(z)
    for step in range(1,len(y)):
        z = oplus(z,y[step])
        dst.append(z.data)
    return dst

def delta_arrow(x,y,rad,delta=0.01):
    X = x - delta*np.cos(rad)
    Y = y - delta*np.sin(rad)
    U = x - X
    V = y - Y
    return X,Y,U,V

if __name__=='__main__':
# read log
    dir_log = sys.argv[1]
    log_pos = dir_log+'/log_pos.csv'
    log_pos_t = dir_log+'/log_pos_t.csv'
    log_x = dir_log+'/log_x.csv'
    log_v = dir_log+'/log_v.csv'
    log_w = dir_log+'/log_w.csv'
    log_path = dir_log+'/log_path.csv'
    path = data.read_path_csv(log_path)
    pos = data.read_path_csv(log_pos)
    pos_t = data.read_path_csv(log_pos_t)
    x = pd.read_csv(log_x,header=None).values
    x = np.reshape(x,(len(x),10,2))
    v = pd.read_csv(log_v,header=None).values
    w = pd.read_csv(log_w,header=None).values
    pad = np.zeros((len(x),10),dtype=np.float32)
    y_pad = np.stack([v,pad,w],axis=2)
# figure
    gs = GridSpec(nrows=3,ncols=2,height_ratios=[1,1,1])
    gs_tmp1 = GridSpecFromSubplotSpec(nrows=3,ncols=1,subplot_spec=gs[0:3,0:1])
    gs_tmp2 = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs[2:3,1:2])
    gs_tmp3 = GridSpecFromSubplotSpec(nrows=2,ncols=1,subplot_spec=gs[0:2,1:2])
    fig = plt.figure(figsize=(12,6))
    #
    ax1 = fig.add_subplot(gs_tmp1[:,:])
    ax1.plot(path[:,0],path[:,1],c='red',linewidth=2)
    ax1.plot(pos[:,0],pos[:,1],c='green' ,linewidth=2)
    ax1.plot(pos_t[:,0],pos_t[:,1],c='blue' ,linewidth=2)
    p1_pos, = ax1.plot(0.0, 0.0,'o',c='black',markersize=10)
    #p1_mouse, = ax1.plot(0.0, 0.0,'o',markersize=10)
    #ln1_mouse, = ax1.plot([],[])
    #ax1.set_position([0.1,0.2,0.3,0.4])
    plt.connect('motion_notify_event',onClick)
    #
    ax2 = fig.add_subplot(gs_tmp2[:,:])
    z = np.array(calc_oplus(y_pad[0]))
    X = z[0:len(z)-1,0]
    Y = z[0:len(z)-1,1]
    U = z[1:len(z),0] - X
    V = z[1:len(z),1] - Y
    U = np.insert(U,0,X[0])
    V = np.insert(V,0,Y[0])
    X = np.insert(X,0,0.0)
    Y = np.insert(Y,0,0.0)
    p2_arrow = ax2.quiver(X,Y,U,V, scale_units='xy', angles='xy', scale=1,color='blue', width=0.005)
    p2_waypoint, = ax2.plot(x[0,:-1,0],x[0,:-1,1],c="red",marker="o",markersize=8,linewidth=0)
    ax2.set_xlim([0,0.4])
    ax2.set_ylim([-0.1,0.1])
    ax2.grid()
    # ax2.legend(['velocity(output)','waypoints(input)'])
    #
    ax3 = fig.add_subplot(gs_tmp3[:,:])
    ax3.plot(path[:,0],path[:,1],c='red',linewidth=2)
    ax3.plot(pos[:,0],pos[:,1],c='green' ,linewidth=2)
    ax3.plot(pos_t[:,0],pos_t[:,1],c='blue' ,linewidth=2)
    #ax3.plot(path[:,0],path[:,1],'o',c='red',mec='red',linewidth=2)
    #ax3.plot(pos[:,0],pos[:,1],'o',c='blue' ,linewidth=2)
    delta = 0.02
    X,Y,U,V = delta_arrow(pos_t[0,0],pos_t[0,1],pos_t[0,2],delta=delta)
    p3_selfpos = ax3.quiver(X,Y,U,V, scale_units='xy', angles='xy', scale=1,color='black',width=delta)
    #X,Y,U,V = delta_arrow(path[0,0],path[0,1],pos[0,2],delta=delta)
    #p3_truepos = ax3.quiver(X,Y,U,V, scale_units='xy', angles='xy', scale=1,color='red',width=delta)
    ax3.legend(['Waypoints','Odom','ModelState','robot'])
    radius = 0.1
    ax3.set_xlim([pos_t[0,0]-radius,pos_t[0,0]+radius])
    ax3.set_ylim([pos_t[0,1]-radius,pos_t[0,1]+radius])
    cid = fig.canvas.mpl_connect('key_press_event',onPressed)
    frame_id = 0
    plt.show()

