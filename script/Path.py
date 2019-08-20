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
    rate = 10 # [Hz]
    point_interval = 50.0 # [mm]
    num_step = 10
    # ROS
    rcp = ROSCartPath()
    rate = rospy.Rate(rate);
    # TEST PATH
    interval = point_interval / 1000.0
    testpath = data.read_path_csv("./pathData/sample_path.csv")
    testpath = testpath
    near_idx = 0
    try:
        while not rospy.is_shutdown():
            cart_pose = quaternion_to_euler(rcp.selfpose.orientation)
            pos_rad = cart_pose.z
            cartpos = settings.xp.array((rcp.selfpose.position.x, rcp.selfpose.position.y, pos_rad),settings.xp.float32)
            #
            idx = data.get_nearly_point_idx(testpath,cartpos)
            testpath_es, idx_list = data.get_evenly_spaced_points(testpath[idx::],interval)
            if len(testpath_es) < num_step+1:
                continue;
            input_path_global = testpath_es[1:num_step+1]
            input_path_local = coordinate.globalpos_to_localpos(input_path_global,cartpos)
            # print(input_path_local)
            rcp.set_path_full(testpath[:,0:2])
            path_plan = settings.xp.vstack((cartpos,input_path_global))
            rcp.set_path_input(path_plan[:,0:2])
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    # sys.exit()

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

class ROSCartPath:
    def __init__(self):
        rospy.init_node('cart_path', anonymous=True)
        init_pose = PoseStamped()
        self.offset = 0.0
        self.selfpose = init_pose.pose
        rospy.Subscriber('/odom',Odometry,self.get_position)
        self.pub_path = rospy.Publisher('/cart/path',Path,queue_size=10)
        self.pub_path_input = rospy.Publisher('/cart/path_input',Path,queue_size=10)
    def set_path_full(self, points):
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
    def set_path_input(self, points):
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
        self.pub_path_input.publish(path)
    def get_position(self, odom):
        self.selfpose = odom.pose.pose
        self.selfpose.position.x = self.selfpose.position.x - self.offset
        self.selfpose.position.y = self.selfpose.position.y - self.offset

if __name__=='__main__':
    settings.set_gpu(-1)
    main()
