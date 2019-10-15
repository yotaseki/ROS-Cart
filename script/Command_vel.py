import rospy
import sys
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Vector3
from nav_msgs.msg import Path, Odometry
from chainer import Variable
from chainer import initializers, serializers
import tf
import xp_settings as settings
import data
import coordinate
from model import Oplus, Generator
import time

def main():
    # Params
    hz = 10
    num_waypoint = 10
    velocity = 0.3
    max_v = 0.5
    max_w = xp.pi * 0.5
    num_step = num_waypoint
    waypoint_interval = velocity / hz
    # ROS Settings
    rospy.init_node('CartController', anonymous=True)
    controller = Controller()
    navigator = Navigator(num_waypoint,waypoint_interval)
    navigator.read_path_csv(sys.argv[1], scale=1.0)
    rate = rospy.Rate(hz);
    # LOAD WEIGHT
    weight = sys.argv[2]
    model = Generator(num_waypoint, num_step)
    serializers.load_npz(weight, model)
    t_navi = time.time()
    t_com = time.time()
    t_all = time.time()
    try:
        while not rospy.is_shutdown():
            t_all = time.time()
            if(navigator.ready):
                t_navi= time.time()
                x = navigator.step()
                t_navi= time.time() - t_navi
                if len(x) == num_step:
                    t_com = time.time()
                    print('\nodom')
                    print(navigator.selfpos)
                    print('input[x,y]')
                    print(x[:,0:2])
                    x = xp.array([x[:,0:2].flatten()], dtype=xp.float32)
                    x = Variable(x)
                    y = model(x)
                    print('output[v,w]')
                    print(y.data[0])
                    params = y.data[0]
                    v = xp.clip(params[0,0] * hz, -max_v,max_v)
                    w = xp.clip(params[0,1] * hz, -max_w, max_w)
                    print('Command')
                    print(v,w)
                    controller.command_vel(v,w)
                    t_com = time.time() - t_com
                    if(xp.sqrt(xp.sum(x.data[0,0:2]**2))) > waypoint_interval*10:
                        print('failed...')
                        break
                else:
                    print('finished')
                    break
            t_all = time.time() - t_all
            print('processing time: ',t_all,'[sec]')
            print('|- Navigator time: ',t_navi,'[sec]')
            print('|- Controller time: ',t_com,'[sec]')
            rate.sleep()
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

    def update_selfpose(self, odom):
        self.selfpose = odom.pose.pose
        # self.selfpose.position.x = self.selfpose.position.x - self.pose_offset_x
        # self.selfpose.position.y = self.selfpose.position.y - self.pose_offset_y
        self.ready = True

if __name__=='__main__':
    settings.set_gpu(-1)
    xp = settings.xp
    main()
