import rospy
import chainer
from chainer import Function, FunctionNode, Link, Chain, ChainList, Variable, optimizers, iterators
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
import numpy as np
from chainer.cuda import cupy as cp

def main():
    train_nes()

def train_nes():
    def Error(xt, x):
        w = cp.array([1., 1., 0.],dtype=cp.float32)
        t = xt * w
        p = x * w
        e = F.sum((t-p)**2)
        return e
    def Command(v, w, draw_path=True):
        z = []
        selfpos = navigator.get_position3D(navigator.selfpose_t)
        z.append(selfpos)
        v_sec = v * options.DATA_HZ
        w_sec = w * options.DATA_HZ
        if(draw_path==True):
            x_gl = coordinate.localpos_to_globalpos(x_data,selfpos)
            msg = navigator.xparray_to_nav_msgs(x_gl[:,0:2])
            navigator.display_path(msg)
        for step in range(len(v[0])):
            controller.command_vel(v_sec[0,step], w_sec[0,step])
            rate.sleep()
            selfpos = navigator.get_position3D(navigator.selfpose_t)
            z.append(selfpos)
        z = cp.array(z, dtype=cp.float32)
        z = coordinate.globalpos_to_localpos(z[1:],z[0])
        return z
    def perturbation(shape, sigma=0.01):
        R, dim = shape
        mu_arr = np.zeros(dim,dtype=np.float32)
        sigma_arr = np.eye((dim),dtype=np.float32) * sigma **2
        ret = np.random.multivariate_normal(mu_arr,sigma_arr,R).astype(np.float32)
        return ret
    def padding(v,w):
        pad = cp.zeros((1,len(v[0])),dtype=v.dtype)
        y = cp.stack((v,pad,w),axis=2)
        return y
    class SimulatorNES(function_node.FunctionNode):
        def forward_gpu(self, inputs):
            v, w = inputs
            if len(v.data)!=len(w.data):
                print('Error: inputs error {} != {}').format(t1,t2)
                raise Exception
            self.R = 1
            self.sigma = .01
            self.p = cp.asarray(perturbation((self.R,2),sigma=self.sigma))
            v_up = v + self.p[:,0]
            v_down = v - self.p[:,0]
            w_up = w + self.p[:,1]
            w_down = w - self.p[:,1]
            # send command
            Command(v_wait, w_wait, draw_path=False)
            z_up = Command(v_up, w_up)
            Command(v_wait, w_wait, draw_path=False)
            z_down = Command(v_down, w_down)
            z = cp.array( [z_up, z_down],dtype=cp.float32)
            self.retain_inputs([0,1])
            self.retain_outputs([0])
            return z,
        def backward(self, inputs, grad_outputs):
            gy, = grad_outputs
            v, w = self.get_retained_inputs()
            v = np.array(v.data,dtype=cp.float32)
            w = np.array(w.data,dtype=cp.float32)
            z_up, z_down = self.get_retained_outputs()[0]
            z_up = cp.array(z_up.data, dtype=cp.float32)
            z_down = cp.array(z_down.data, dtype=cp.float32)
            D = z_up - z_down
            dv = D * self.p[:,0] * 0.5 / (self.sigma**2 * self.R)
            gv_up = F.sum(gy[0] * dv, axis=1)
            gv_down = F.sum(gy[1] * dv, axis=1)
            gv = F.expand_dims(gv_up + gv_down, axis=0)
            dw = D * self.p[:,1] * 0.5 / (self.sigma**2 * self.R)
            gw_up = F.sum(gy[0] * dw, axis=1)
            gw_down = F.sum(gy[1] * dw, axis=1)
            gw =F.expand_dims(gw_up + gw_down, axis=0)
            return gv,gw
    def sim_nes(v,w):
        return SimulatorNES().apply((v,w))[0]
    options.init()
    rospy.init_node('CartController',anonymous=True)
    controller = Controller()
    navigator = Navigator(options.DATA_NUM_WAYPOINTS, options.DATA_V_STEP)
    rate = rospy.Rate(options.DATA_HZ)
    model = Generator(options.DATA_NUM_WAYPOINTS,options.DATA_NUM_PREVIOUS_U, options.DATA_NUM_STEP)
    serializers.load_npz(argv[1],model)
    opt = optimizers.Adam()
    opt.setup(model)
    serializers.load_npz(argv[2],opt)
    if settings.gpu_index >= 0:
        cuda.cupy.cuda.Device(settings.gpu_index).use()
        model.to_gpu(settings.gpu_index)
    X = train_tools.make_dataset()
    random.shuffle(X)
    max_epoch = 10
    epoch = 0
    itr = 0
    v_wait = xp.ones((1,10), dtype=xp.float32) * options.DATA_V_STEP
    w_wait = xp.zeros((1,10), dtype=xp.float32)
    loss = .0
    loss_epoch = []
    #dirname = 'simNES_' + os.path.splitext(argv[1])[0]
    dirname = 'NES_' + os.path.dirname(argv[1])
    os.mkdir(dirname)
    try:
        while not rospy.is_shutdown():
            idx = counter % options.DATA_SIZE
            x_data = xp.array(X[idx][:],dtype=xp.float32)
            x = xp.ravel(x_data[:,0:2])
            x = xp.array([x],dtype=xp.float32)
            x = Variable(x)
            selfpos = navigator.get_position3D(navigator.selfpose_t)
            uv,uw = model(x)
            v_lim = options.DATA_MAX_V_STEP
            w_lim = options.DATA_MAX_W_STEP
            v = F.clip(uv,.0,v_lim)
            w = F.clip(uw,-w_lim,w_lim)
            z = simnes(v,w)
            mz = (z[0]+z[1]) * 0.5
            e = Error(x_data, mz)
            model.cleargrads()
            e.backward()
            opt.update()
            itr = itr + 1
            print('itr:', itr, ' D:',xp.sum(e.data))
            if counter % options.DATA_SIZE == 0 and counter != 0:
                epoch = epoch + 1
                msg = '******Epoch:' +  str(epoch)
                print(msg)
                log_dmesg.append(msg)
                msg = 'D_average:' + str(D_loss / options.DATA_SIZE)
                print(msg)
                log_dmesg.append(msg)
                D_loss = .0
                serializers.save_npz(dirname+'/'+dirname+'ep'+str(epoch)+'.model', model)
                serializers.save_npz(dirname+'/'+dirname+'ep'+str(epoch)+'.state', opt)
                str_ = '\n'.join(log_dmesg)
                log_dmesg = []
                with open(dirname+"/loss.txt", "a") as f:
                    f.write(str_)
            if epoch == max_epoch:
                serializers.save_npz(dirname+'/'+dirname+'.model', model)
                serializers.save_npz(dirname+'/'+dirname+'.state', opt)
                break
    except rospy.ROSInterruptException:
        sys.exit()

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
