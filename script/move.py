import rospy
import sys
from model import Oplus, Generator
from geometry_msgs.msg import Twist
import numpy as np
from chainer.cuda import cupy as cp
from chainer import Variable
from chainer import initializers, serializers

def main():
    # ROS
    roscart = ROSCart()
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
        print("x",x)
        print("y",y.data[0])
        roscart.move(y.data[0])
        # roscart.move(xp.array([[1.0,0.0], [1.0,0.0],[1.0,0.0]]))
        #while not rospy.is_shutdown():
        #    pass
    except rospy.ROSInterruptException:
        pass
    sys.exit()

def forward(model, x):
    x = Variable(x)
    y = model(x)
    return y

class ROSCart:
    def __init__(self):
        rospy.init_node('move', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=10)
        self.twist = Twist()
        self.rate = rospy.Rate(0.2); # 1sec
    def move(self, params):
        for v,w in params:
            print [v, w]
            self.twist.linear.x = v
            self.twist.angular.z = w
            self.pub.publish(self.twist)
            self.rate.sleep()

if __name__=='__main__':
    gpu_idx = -1
    if(gpu_idx >= 0):
        xp = cp
    else:
        xp = np
    main()
