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
    hz = 100
    num_waypoint = 10
    num_step = num_waypoint
    waypoint_interval = 0.05
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
    try:
        while not rospy.is_shutdown():
            if(navigator.ready):
                x = navigator.step()
                if len(x) == num_step:
                    print('')
                    print('input[x,y]')
                    print(x[:,0:2])
                    x = xp.array([x[:,0:2].flatten()], dtype=xp.float32)
                    x = Variable(x)
                    y = model(x)
                    print('output[v,w]')
                    print(y.data[0])
                    params = y.data[0]
                    v = params[0,0] * 10 # * 0.5
                    w = params[0,1] * 10 # * 0.5
                    #if(xp.abs(v) > 0.5 or xp.abs(w) > xp.pi):
                    #    break
                    controller.command_vel(v,w)
                else:
                    print('finished')
                    break
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    # sys.exit()

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
        self.pub_path = rospy.Publisher('/cart/path',Path,queue_size=1)
        self.pub_input = rospy.Publisher('/cart/input',Path,queue_size=1)
        self.selfpose = PoseStamped().pose
        self.num_waypoint = num_waypoint
        self.waypoint_interval = waypoint_interval
        self.path = []
        self.ready = False

    def read_path_csv(self, filename, scale=1.0):
        self.path = data.read_path_csv(filename)
        self.path = self.path * scale

    def step(self):
        pose = self.quaternion_to_euler(self.selfpose.orientation)
        x = self.selfpose.position.x
        y = self.selfpose.position.y
        th = pose.z
        state = xp.array((x,y,th),xp.float32)
        idx = data.get_nearly_point_idx(self.path, state)
        x_data_gl = data.get_waypoints(idx, self.path, self.num_waypoint, self.waypoint_interval)
        if len(x_data_gl) < self.num_waypoint:
            self.display_input([])
            return []
        disp_interbal = int(len(self.path)/100)
        self.display_path(self.path[::disp_interbal,0:2])
        self.display_input(xp.vstack((state,x_data_gl))[:,0:2])
        x_data = coordinate.globalpos_to_localpos(x_data_gl, state)
        return x_data

    def quaternion_to_euler(self, quaternion):
        e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return Vector3(x=e[0], y=e[1], z=e[2])

    def display_path(self, points):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = rospy.Time.now()
        for x,y in points:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub_path.publish(path)

    def display_input(self, points):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = rospy.Time.now()
        for x,y in points:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub_input.publish(path)

    def update_selfpose(self, odom):
        self.selfpose = odom.pose.pose
        # self.selfpose.position.x = self.selfpose.position.x - self.pose_offset_x
        # self.selfpose.position.y = self.selfpose.position.y - self.pose_offset_y
        self.ready = True

if __name__=='__main__':
    settings.set_gpu(-1)
    xp = settings.xp
    main()
