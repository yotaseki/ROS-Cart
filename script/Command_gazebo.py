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
CAPTURE_LOG = True

def main():
    print('PATH:',sys.argv[1])
    print('WEIGHT:',sys.argv[2])
    # Params
    hz = 10
    num_waypoint = 10
    num_step = num_waypoint
    v_sec = 0.3
    max_v = 0.5
    max_w = xp.pi * 0.5
    waypoint_interval = v_sec / hz
    # ROS Settings
    path_name = sys.argv[1]
    rospy.init_node('CartController', anonymous=True)
    controller = Controller()
    navigator = Navigator(num_waypoint,waypoint_interval)
    navigator.read_path_csv(path_name, scale=1.0)
    rate = rospy.Rate(hz);
    # LOAD WEIGHT
    weight_name = sys.argv[2]
    model = Generator(num_waypoint, num_step)
    serializers.load_npz(weight_name, model)
    t_navi = time.time()
    t_com = time.time()
    t_all = time.time()

    if CAPTURE_LOG:
        dir_log = 'GazeboLog_'+os.path.basename(weight_name)+'_'+os.path.basename(path_name)
        os.mkdir(dir_log)
        log_pos = []
        log_x = []
        log_v = []
        log_w = []
    du = xp.array([.0,.0],dtype=xp.float32)
    try:
        while not rospy.is_shutdown():
            t_all = time.time()
            if(navigator.ready):
                t_navi= time.time()
                x_data = navigator.step()
                selfpos = navigator.get_position3D(navigator.selfpose)
                selfpos_t = navigator.get_position3D(navigator.selfpose_t)
                t_navi= time.time() - t_navi
                if len(x) == num_step:
                    t_com = time.time()
                    #print('\nodom')
                    #print(navigator.selfpos)
                    #print('input[x,y]')
                    #print(x[:,0:2])
                    x = xp.vstack((x_data[:,0:2],du))
                    x = xp.ravel(x)
                    x = xp.array([x],dtype=xp.float32)
                    x = Variable(x)
                    uv,uw = model(x)
                    v_lim = options.DATA_MAX_V_STEP
                    w_lim = options.DATA_MAX_W_STEP
                    v = xp.clip(uv,.0,v_lim)
                    w = xp.clip(uw,-w_lim,w_lim)
                    com_v = uv[0,0]* hz
                    com_w = uw[0,0]* hz
                    du[0] = uv[0,0]
                    du[1] = uw[0,0]
                    #print('output[v,w]')
                    #print(v.data)
                    #print(w.data)
                    #print('Command')
                    print(com_v,com_w)
                    controller.command_vel(com_v,com_w)
                    t_com = time.time() - t_com
                    if CAPTURE_LOG:
                        log_pos.append(selfpos)
                        log_pos_t.append(selfpos_t)
                        log_x.append(x.data[0])
                        log_v.append(v.data[0,:])
                        log_w.append(w.data[0,:])
                    #if(xp.sqrt(xp.sum(x.data[0,0:2]**2))) > waypoint_interval*10:
                    #    print('failed...')
                    #    break
                else:
                    print('finished')
                    break
            rate.sleep()
            t_all = time.time() - t_all
            print('time: ',t_all,'[sec]')
            #print('|- Navigator time: ',t_navi,'[sec]')
            #print('|- Controller time: ',t_com,'[sec]')
        if CAPTURE_LOG:
            list_to_csv(log_pos,dir_log+'/log_pos.csv')
            list_to_csv(log_pos_t,dir_log+'/log_pos_t.csv')
            list_to_csv(log_x,dir_log+'/log_x.csv')
            list_to_csv(log_v,dir_log+'/log_v.csv')
            list_to_csv(log_w,dir_log+'/log_w.csv')
            data.write_path_csv(navigator.path, dir_log+'/log_path.csv')
    except rospy.ROSInterruptException:
        sys.exit()

def list_to_csv(l,name):
    arr = xp.array(l,dtype=xp.float32)
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

    def step(self):
        selfpos = navigator.get_position3D(self.selfpose_t)
        #t0 = time.time()
        idx = data.get_nearly_point_idx(self.path, selfpos)
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

if __name__=='__main__':
    settings.set_gpu(-1)
    xp = settings.xp
    main()
