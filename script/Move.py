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
    hz = 10 # [Hz]
    point_interval = 100.0 # [mm]
    # ROS
    roscart = ROSCart()
    rate = rospy.Rate(hz);
    # MODEL
    num_step = 10
    num_waypoint = num_step
    model = Generator(num_waypoint, num_step)
    # LOAD WEIGHT
    weight = sys.argv[1]
    serializers.load_npz(weight, model)
    # TEST PATH
    try:
        while not rospy.is_shutdown():
            if(roscart.ready):
                input_path = roscart.nextpath()
                print('')
                print('input[x,y]')
                print(input_path[:,0:2])
                x = xp.array([input_path[:,0:2].flatten()], dtype=xp.float32)
                t_start= time.time()
                x = Variable(x)
                y = model(x)
                t_elapsed = time.time() - t_start
                print('output[v,w]')
                print(y.data[0])
                print('time:',t_elapsed,'[sec]')
                params = y.data[0]
                v = params[0,0] * hz
                w = params[0,1] * hz
                roscart.command_vel(v,w)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    # sys.exit()

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

class ROSCart:
    def __init__(self):
        rospy.init_node('velo', anonymous=True)
        init_pose = PoseStamped()
        self.ready = False
        self.next = 0
        self.offset = 0.0
        self.selfpose = init_pose.pose
        self.pub_twi = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=10)
        rospy.Subscriber('/cart/path_input',Path,self.set_next_path)
        rospy.Subscriber('/odom',Odometry,self.get_position)
    def command_vel(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub_twi.publish(twist)
    def nextpath(self):
        return self.next
    def set_next_path(self, path):
        poses = xp.empty((0,3),dtype=xp.float32)
        for i in range(len(path.poses)):
            pos = path.poses[i].pose.position
            pos_ori = quaternion_to_euler(path.poses[i].pose.orientation)
            pose = xp.array([[pos.x,pos.y,pos_ori.z]],dtype=xp.float32)
            poses = xp.vstack((poses, pose))
        cart_ori = quaternion_to_euler(self.selfpose.orientation)
        cartpos = xp.array((self.selfpose.position.x, self.selfpose.position.y, cart_ori.z),xp.float32)
        path_local = coordinate.globalpos_to_localpos(poses[1:],cartpos)
        self.next = path_local[:,0:2]
        self.ready = True
    def get_position(self, odom):
        self.selfpose = odom.pose.pose
        self.selfpose.position.x = self.selfpose.position.x - self.offset
        self.selfpose.position.y = self.selfpose.position.y - self.offset

if __name__=='__main__':
    settings.set_gpu(-1)
    xp = settings.xp
    main()
