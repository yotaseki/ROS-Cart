import rospy
import sys
from model import Oplus, Generator
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
import numpy as np
from chainer.cuda import cupy as cp
from chainer import Variable
from chainer import initializers, serializers
from data import PathData

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
    k = xp.pi / 6   # curvature
    testpath = pdata.make_arc_path_2(3, k)
    div_testpath = pdata.get_n_point_from_path_2(num_step, testpath)
    x = xp.array([div_testpath[:,0:2].flatten()], dtype=np.float32)
    try:
        while not rospy.is_shutdown():
            y = forward(model, x)
            params = y.data[0]
            v = params[0,0]
            w = params[0,1]
            roscart.move(v,w)
            roscart.set_path_input(testpath[:,0:2])
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
            pose.pose.position.x = # x + start_pose.position.x
            pose.pose.position.y = # y + start_pose.position.y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub_path_input.publish(path)
    def get_position(self, amcl_base):
        self.selfpose = amcl_base.pose.pose

if __name__=='__main__':
    main()
