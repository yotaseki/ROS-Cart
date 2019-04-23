import rospy
from geometry_msgs.msg import Twist

def main():
    rospy.init_node('move', anonymous=True)
    cart = Cart()
    try:
        cart.move()
    except rospy.ROSInterruptException:
        pass

class Cart:
    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=10)
        self.twist = Twist()
        self.rate = rospy.Rate(10); # 0.1sec
    def move(self):
        while not rospy.is_shutdown():
            self.twist.linear.x = 1.0
            self.twist.angular.z = 1.0
            self.pub.publish(self.twist)
            self.rate.sleep()

if __name__=='__main__':
    main()
