import rospy
import chainer
from chainer import Link, Chain, ChainList, Variable, optimizers, iterators
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer import serializers
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Vector3
from nav_msgs.msg import Path, Odometry
from gazebo_msgs.msg import ModelStates
import tf
import sys, os
import random
import time, datetime
import pandas as pd

import options
import xp_settings as settings
import data
import coordinate
import train_tools
from model import Oplus, Generator, calc_oplus
settings.set_gpu(0)
xp = settings.xp

def main():
    print('WEIGHT:',sys.argv[1])
    # Params
    options.init()
    # ROS Settings
    rospy.init_node('CartController', anonymous=True)
    controller = Controller()
    navigator = Navigator(options.DATA_NUM_STEP,options.DATA_V_STEP)
    rate = rospy.Rate(options.DATA_HZ);
    rate.sleep()
    # LOAD WEIGHT
    model_name = sys.argv[1]
    model = Generator(options.DATA_NUM_WAYPOINTS, options.DATA_NUM_PREVIOUS_U, options.DATA_NUM_STEP)
    serializers.load_npz(model_name, model)
    #opt_name = sys.argv[2]
    opt = optimizers.Adam()
    opt.setup(model)
    #serializers.load_npz(opt_name, opt)
    if settings.gpu_index >= 0:
        cuda.cupy.cuda.Device(settings.gpu_index).use()
        model.to_gpu(settings.gpu_index)
    options.DATA_SIZE = 1000
    X = train_tools.make_dataset()
    random.shuffle(X)
    max_epoch = 30
    iterate = 0
    AvgLoss=[]
    avgloss_simu = .0
    avgloss_oplus = .0
    replanning = True
    selfpos = navigator.get_position3D(navigator.selfpose_t)
    v_wait = xp.ones((1,10), dtype=xp.float32) * options.DATA_V_STEP
    w_wait = xp.zeros((1,10), dtype=xp.float32)
    try:
        def Command(v, w, draw_path=True):
            z = []
            selfpos = navigator.get_position3D(navigator.selfpose_t)
            selfpos_odom = navigator.get_position3D(navigator.selfpose)
            z.append(selfpos)
            v_sec = v * options.DATA_HZ
            w_sec = w * options.DATA_HZ
            if(draw_path==True):
                x_gl = coordinate.localpos_to_globalpos(x_data,selfpos_odom)
                msg = navigator.xparray_to_nav_msgs(x_gl[:,0:2])
                navigator.display_path(msg)
            for step in range(len(v[0])):
                controller.command_vel(v_sec[0,step], w_sec[0,step])
                rate.sleep()
                selfpos = navigator.get_position3D(navigator.selfpose_t)
                z.append(selfpos)
            z = xp.array(z, dtype=xp.float32)
            z = coordinate.globalpos_to_localpos(z[1:],z[0])
            return z
        while not rospy.is_shutdown():
            t_all = time.time()
            if(navigator.ready):
                t_navi= time.time()
                idx = iterate%options.DATA_SIZE
                x_data = xp.array(X[idx][:],dtype=xp.float32)
                x = xp.ravel(x_data[:,0:2])
                x = xp.array([x],dtype=xp.float32)
                x = Variable(x)
                uv,uw = model(x)
                v_lim = options.DATA_MAX_V_STEP
                w_lim = options.DATA_MAX_W_STEP
                v = F.clip(uv,.0,v_lim)
                w = F.clip(uw,-w_lim,w_lim)
                #print('pose')
                #print(navigator.selfpos)
                #print('input[x,y]')
                #print(x)
                #print('output[v,w]')
                #print(v.data)
                #print(w.data)
                pad = Variable(xp.zeros((1,options.DATA_NUM_STEP),dtype=xp.float32))
                y = F.stack((v,pad,w),axis=2)
                z_oplus = calc_oplus(y[0])
                Command(v_wait, w_wait, draw_path=False)
                z_simu = Command(v.data, w.data)
                z_t = x_data
                simu_loss = train_tools.loss_function(z_simu,z_oplus)
                oplus_loss = train_tools.loss_function(z_oplus,z_t)
                loss = simu_loss + oplus_loss
                avgloss_oplus = avgloss_oplus + oplus_loss.data
                avgloss_simu = avgloss_simu + simu_loss.data
                model.cleargrads()
                loss.backward()
                opt.update()
                print('iter: ',iterate, '  Loss(Oplus):',oplus_loss.data, ' Loss(Simu):',simu_loss.data)
                iterate = iterate + 1
                if((iterate%options.DATA_SIZE)==0):
                    epoch = int(iterate/options.DATA_SIZE)
                    print('Epoch:',epoch, '    AvgLoss(Oplus):',avgloss_oplus/len(X), '    AvgLoss(Simu)',avgloss_simu/len(X))
                    random.shuffle(X)
                    AvgLoss.append([avgloss_oplus/len(X), avgloss_simu/len(X)])
                    avgloss_simu = .0
                    avgloss_oplus = .0
                    serializers.save_npz('Advanced'+str(epoch)+'ep_'+os.path.basename(model_name), model)
                    serializers.save_npz('Advanced'+str(epoch)+'ep_'+os.path.basename(model_name), opt)
                if int(iterate/options.DATA_SIZE)==max_epoch:
                    print('finished')
                    print('AvgLoss = ',AvgLoss)
                    break
            t_all = time.time() - t_all
            #print('time: ',t_all,'[sec]')
            #print('|- Navigator time: ',t_navi,'[sec]')
            #print('|- Controller time: ',t_com,'[sec]')
    except rospy.ROSInterruptException:
        sys.exit()

def list_to_csv(l,name):
    arr = xp.array(l,dtype=xp.float32)
    print(name,'->',arr.shape)
    df = pd.DataFrame(arr)
    df.to_csv(name,index=False,header=False)

class Controller:
    def __init__(self):
        self.pub_twi = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=1)
    def command_vel(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub_twi.publish(twist)

class Navigator:
    def __init__(self, num_waypoint, waypoint_interval):
        rospy.Subscriber('/odom',Odometry,self.update_selfpose)
        rospy.Subscriber('/gazebo/model_states',ModelStates,self.update_selfpose_t)
        self.pub_path = rospy.Publisher('/cart/path',Path,queue_size=5)
        self.pub_input = rospy.Publisher('/cart/input',Path,queue_size=5)
        self.selfpose = PoseStamped().pose
        self.selfpose_t = PoseStamped().pose
        self.num_waypoint = num_waypoint
        self.waypoint_interval = waypoint_interval
        self.path = []
        self.path_nav_msg = None
        self.ready = False

    def read_path_csv(self, filename, scale=1.0):
        self.path = data.read_path_csv(filename)
        self.path = self.path * scale
        disp_interbal = int(len(self.path)/100)
        self.path_nav_msg = self.xparray_to_nav_msgs(self.path[::disp_interbal,0:2])

    def sampling(self,rad,translate=0,rotate=0):
        d = data.generate_arc_path(self.num_waypoint,rad,self.waypoint_interval)
        #print(self.num_waypoint, rad, self.waypoint_interval)
        #print(d)
        d = data.rotate_path(d,rad*0.5)
        if translate != 0:
            rand_trans_x = xp.random.rand() * translate
            d = data.translate_path(d,rand_trans_x,0.0)
        if rotate != 0:
            rand_rotate = xp.random.rand()*(rotate*2)-rotate
            d = data.rotate_path(d,rand_rotate)
        x_lc = d
        selfpos = self.get_position3D(self.selfpose_t)
        x_gl = coordinate.localpos_to_globalpos(x_lc, selfpos)
        msg = self.xparray_to_nav_msgs(x_gl[:,0:2])
        self.display_path(msg)
        return x_lc

    def step(self):
        selfpos = self.get_position3D(self.selfpose)
        #t0 = time.time()
        idx = data.get_nearly_point_idx(self.path, self.selfpos)
        #print('t0',time.time() - t0)
        #t1 = time.time()
        x_data_gl = data.get_waypoints(idx, self.path, self.num_waypoint, self.waypoint_interval)
        #print('t1',time.time() - t1)
        #t2 = time.time()
        if len(x_data_gl) < self.num_waypoint:
            self.display_path(self.path_nav_msg)
            self.display_input(Path())
            return []
        self.display_path(self.path_nav_msg)
        input_nav_msg = self.xparray_to_nav_msgs(xp.vstack((selfpos,x_data_gl))[:,0:2])
        self.display_input(input_nav_msg)
        x_data = coordinate.globalpos_to_localpos(x_data_gl, selfpos)
        #print('t2',time.time() - t2)
        return x_data

    def quaternion_to_euler(self, quaternion):
        e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return Vector3(x=e[0], y=e[1], z=e[2])

    def xparray_to_nav_msgs(self, array):
        path = Path()
        path.header.frame_id = ''
        path.header.stamp = rospy.Time.now()
        for x,y in array:
            pose = PoseStamped()
            pose.header.frame_id = ''
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def display_path(self, msg):
        msg.header.frame_id = 'odom'
        msg.header.stamp = rospy.Time.now()
        self.pub_path.publish(msg)

    def display_input(self, msg):
        msg.header.frame_id = 'odom'
        msg.header.stamp = rospy.Time.now()
        self.pub_input.publish(msg)
    
    def get_position3D(self,pose):
        eu = self.quaternion_to_euler(pose.orientation)
        x = pose.position.x
        y = pose.position.y
        th = eu.z
        pos3D = xp.array((x,y,th),xp.float32)
        return pos3D

    def update_selfpose(self, odom):
        self.selfpose = odom.pose.pose
        self.ready = True

    def update_selfpose_t(self,states):
        idx  = states.name.index('mobile_base')
        self.selfpose_t = states.pose[idx]
        self.ready = True

if __name__=='__main__':
    main()
