import rospy
import sys
from model import Oplus, Generator
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Vector3
from nav_msgs.msg import Path
import numpy as np
from chainer.cuda import cupy as cp
from chainer import Variable
from chainer import initializers, serializers
from data import PathData
import tf

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

def global2local(path, pos):
    pos_x = pos[0]
    pos_y = pos[1]
    pos_rad = pos[2]
    X = path[:,0]
    Y = path[:,1]
    R = path[:,2]
    X = (X * np.cos(pos_rad) - Y * np.sin(pos_rad)) + pos_x
    Y = (X * np.sin(pos_rad) + Y * np.cos(pos_rad)) + pos_y
    R = R + pos_rad
    ret = np.array((X,Y,R),np.float32)
    return ret

def main():
    gpu_idx = -1
    if(gpu_idx >= 0):
        xp = cp
    else:
        xp = np
    # ROS
    roscart = ROSCart()
    rate = rospy.Rate(10); # 0.1sec
    # MODEL
    num_step = 3
    input_dim = num_step*2
    model = Generator(input_dim, num_step)
    # LOAD WEIGHT
    weight = sys.argv[1]
    serializers.load_npz(weight, model)
    # TEST PATH
    # points = xp.array([[1.0,0.0],[2.0,0.0],[3.0,0.0]])
    pdata = PathData(gpu_idx)
    k = xp.pi / 18   # curvature
    testpath = pdata.make_arc_path_2(10, k)
    testpath_es, idx_list = pdata.get_evenly_spaced_points(testpath,1.0)
    near_idx = 0
    try:
        while not rospy.is_shutdown():
            if(near_idx > len(testpath_es)-num_step):
                print('finished')
                break;
            cart_pose = quaternion_to_euler(roscart.selfpose.orientation)
            pos_rad = cart_pose.z
            cartpos = np.array((roscart.selfpose.position.x, roscart.selfpose.position.y, pos_rad),np.float32)
            #
            idx = pdata.get_nearly_point_idx(testpath,cartpos)
            near_idx = pdata.get_next_path_idx(idx,idx_list)
            input_path = testpath_es[near_idx:near_idx+num_step]
            print('')
            print(input_path)
            # input_path = global2local(input_path,cartpos)
            # print(cartpos)
            print(input_path)
            #
            x = xp.array([input_path[:,0:2].flatten()], dtype=np.float32)
            y = forward(model, x)
            params = y.data[0]
            v = params[0,0]
            w = params[0,1]
            roscart.move(v,w)
            roscart.set_path_input(testpath_es[near_idx:near_idx+num_step,0:2])
            roscart.set_path_full(testpath[:,0:2])
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    # sys.exit()

def forward(model, x):
    x = Variable(x)
    y = model(x)
    return y

class ROSCart:
    def __init__(self):
        rospy.init_node('move', anonymous=True)
        init_pose = PoseStamped()
        self.offset = 10.0
        self.selfpose = init_pose.pose
        rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,self.get_position)
        self.pub_path = rospy.Publisher('/cart/path',Path,queue_size=10)
        self.pub_path_input = rospy.Publisher('/cart/path_input',Path,queue_size=10)
        self.pub_twi = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=10)
    def move(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub_twi.publish(twist)
    def set_path_full(self, points):
        start_pose = self.selfpose
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for x,y in points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x # + start_pose.position.x
            pose.pose.position.y = y # + start_pose.position.y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub_path.publish(path)
    def set_path_input(self, points):
        start_pose = self.selfpose
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for x,y in points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x + start_pose.position.x
            pose.pose.position.y = y + start_pose.position.y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub_path_input.publish(path)
    def get_position(self, amcl_base):
        self.selfpose = amcl_base.pose.pose
        self.selfpose.position.x = self.selfpose.position.x - self.offset
        self.selfpose.position.y = self.selfpose.position.y - self.offset

if __name__=='__main__':
    main()
