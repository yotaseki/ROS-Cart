import rospy
import sys, os
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Vector3
from nav_msgs.msg import Path, Odometry
from chainer import Variable
from chainer import initializers, serializers
import tf
import xp_settings as settings
import data
import coordinate
from model import Oplus, Generator
import time, datetime
import pandas as pd
SAMPLING = True

def main():
    print('WEIGHT:',sys.argv[1])
    # Params
    DATA_SIZE = 1000
    DATA_NUM_WAYPOINTS = 10
    DATA_NUM_STEP = DATA_NUM_WAYPOINTS
    DATA_HZ = 10
    DATA_V_STEP = 0.5 / DATA_HZ # [m/step]
    DATA_MAX_V_STEP = 1.0 / DATA_HZ # [m/step]
    DATA_W_STEP = xp.pi * 0.5 / DATA_HZ # [rad/step]
    DATA_MAX_W_STEP = xp.pi * 0.5 / DATA_HZ # [rad/step]
    DATA_RANGE_TRANSLATE = 0
    DATA_RANGE_ROTATE = 0
    v_sec = DATA_V_STEP*DATA_HZ
    max_v = DATA_MAX_V_STEP*DATA_HZ
    max_w = xp.pi * DATA_MAX_W_STEP*DATA_HZ
    # ROS Settings
    rospy.init_node('CartController', anonymous=True)
    controller = Controller()
    navigator = Navigator(DATA_NUM_STEP,DATA_V_STEP)
    rate = rospy.Rate(DATA_HZ);
    # LOAD WEIGHT
    weight_name = sys.argv[1]
    model = Generator(DATA_NUM_WAYPOINTS, DATA_NUM_STEP)
    serializers.load_npz(weight_name, model)
    t_navi = time.time()
    t_com = time.time()
    t_all = time.time()
    itr = 10
    if SAMPLING:
        dir_log = 'DATA_'+os.path.basename(weight_name)
        os.mkdir(dir_log)
    log_x = []
    log_v = []
    log_w = []
    log_pos = []
    x = []
    pos = []
    step = 0
    replanning = True
    try:
        while not rospy.is_shutdown():
            t_all = time.time()
            if(navigator.ready):
                t_navi= time.time()
                if replanning == True:
                    rand_rad = xp.random.rand()*(2*DATA_W_STEP)-DATA_W_STEP
                    p = navigator.sampling(rand_rad)
                    x = xp.array([p[:,0:2].flatten()], dtype=xp.float32)
                    x = Variable(x)
                    y_v,y_w = model(x)
                    #print('\nodom')
                    #print(navigator.selfpos)
                    #print('input[x,y]')
                    #print(p)
                    #print('output[v,w]')
                    #print(y_v.data)
                    #print(y_w.data)
                    replanning = False
                t_navi= time.time() - t_navi
                t_com = time.time()
                v = xp.clip(y_v.data[0,step] * DATA_HZ, 0.0,max_v)
                w = xp.clip(y_w.data[0,step] * DATA_HZ, -max_w, max_w)
                #print('Command')
                #print(v,w)
                navigator.update_selfpos()
                controller.command_vel(v,w)
                t_com = time.time() - t_com
                pos.append(navigator.selfpos)
                step = step + 1
                if step == DATA_NUM_STEP:
                    arr = xp.array(pos)
                    log_x.append(x.data[0])
                    log_v.append(y_v.data[0,:])
                    log_w.append(y_w.data[0,:])
                    log_pos.append(arr.flatten())
                    print('iter: ',len(log_x))
                    step = 0
                    pos = []
                    replanning = True
                    if len(log_x) == itr:
                        print('finished')
                        break
                #if(xp.sqrt(xp.sum(x.data[0,0:2]**2))) > waypoint_interval*10:
                #    print('failed...')
                #    break
            rate.sleep()
            t_all = time.time() - t_all
            #print('time: ',t_all,'[sec]')
            #print('|- Navigator time: ',t_navi,'[sec]')
            #print('|- Controller time: ',t_com,'[sec]')
        if SAMPLING:
            list_to_csv(log_pos,dir_log+'/log_pos.csv')
            list_to_csv(log_x,dir_log+'/log_x.csv')
            list_to_csv(log_v,dir_log+'/log_v.csv')
            list_to_csv(log_w,dir_log+'/log_w.csv')
            data.write_path_csv(navigator.path, dir_log+'/log_path.csv')
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
        # self.pose_offset_x = 0.0
        # self.pose_offset_y = 0.0
        rospy.Subscriber('/odom',Odometry,self.update_selfpose)
        self.pub_path = rospy.Publisher('/cart/path',Path,queue_size=5)
        self.pub_input = rospy.Publisher('/cart/input',Path,queue_size=5)
        self.selfpose = PoseStamped().pose
        self.selfpos = xp.array((0,0,0),xp.float32)
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
        self.update_selfpos()
        x_gl = coordinate.localpos_to_globalpos(x_lc,self.selfpos)
        msg = self.xparray_to_nav_msgs(x_gl[:,0:2])
        self.display_path(msg)
        return x_lc

    def step(self):
        pose = self.quaternion_to_euler(self.selfpose.orientation)
        x = self.selfpose.position.x
        y = self.selfpose.position.y
        th = pose.z
        self.selfpos = xp.array((x,y,th),xp.float32)
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
        input_nav_msg = self.xparray_to_nav_msgs(xp.vstack((self.selfpos,x_data_gl))[:,0:2])
        self.display_input(input_nav_msg)
        x_data = coordinate.globalpos_to_localpos(x_data_gl, self.selfpos)
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
    
    def update_selfpos(self):
        pose = self.quaternion_to_euler(self.selfpose.orientation)
        x = self.selfpose.position.x
        y = self.selfpose.position.y
        th = pose.z
        self.selfpos = xp.array((x,y,th),xp.float32)

    def update_selfpose(self, odom):
        self.selfpose = odom.pose.pose
        # self.selfpose.position.x = self.selfpose.position.x - self.pose_offset_x
        # self.selfpose.position.y = self.selfpose.position.y - self.pose_offset_y
        self.ready = True

if __name__=='__main__':
    settings.set_gpu(-1)
    xp = settings.xp
    main()
