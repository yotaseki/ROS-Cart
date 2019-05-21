import rospy
import sys
from model import Oplus, Generator
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import numpy as np
from chainer.cuda import cupy as cp
from chainer import Variable
from chainer import initializers, serializers

def main():
    # ROS
    roscart = ROSCart()
    rate = rospy.Rate(1); # 1sec
    # MODEL
    num_step = 3
    input_dim = num_step*2
    model = Generator(input_dim, num_step)
    # LOAD WEIGHT
    weight = sys.argv[1]
    serializers.load_npz(weight, model)
    # READ PATH
    points = xp.array([[0.0,1.0],[0.0,2.0],[0.0,3.0]])
    x = xp.array([points.flatten()], dtype=np.float32)
    try:
        y = forward(model, x)
        params = y.data[0]
        # roscart.set_path(points)
        roscart.set_path(xp.array([[0.0,10.0],[10.0,10.0],[10.0,0.0],[0.0,0.0]]))
        for i in range(len(params)):
            v = params[i,0]
            w = params[i,1]
            # roscart.move(v,w)
            rate.sleep()
            # roscart.move(xp.array([[1.0,0.0], [1.0,0.0],[1.0,0.0]]))
        # while not rospy.is_shutdown():
        # rospy.spin()
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
        self.pub_twi = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=10)
        self.pub_path = rospy.Publisher('/cart/path',Path,queue_size=10)
    def move(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub_twi.publish(twist)
    def set_path(self, points):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for x,y in points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
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

if __name__=='__main__':
    gpu_idx = -1
    if(gpu_idx >= 0):
        xp = cp
    else:
        xp = np
    main()
