import rospy
import sys
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Vector3
from nav_msgs.msg import Path
from chainer import Variable
from chainer import initializers, serializers
import tf
import xp_settings as settings
import data
import coordinate
from model import Oplus, Generator
import time

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

def main():
    # Params
    rate = 100 # [Hz]
    point_interval = 100.0 # [mm]
    # ROS
    roscart = ROSCart()
    rate = rospy.Rate(rate); # 10Hz / 0.1sec
    # MODEL
    num_step = 3
    num_waypoint = num_step
    model = Generator(num_waypoint, num_step)
    # LOAD WEIGHT
    weight = sys.argv[1]
    serializers.load_npz(weight, model)
    # TEST PATH
    interval = point_interval / 1000.0
    # points = settings.xp.array([[1.0,0.0],[2.0,0.0],[3.0,0.0]])
    k =  settings.xp.pi / 18   # curvature
    # testpath = data.make_arc_path_2(9, k)
    testpath = data.read_path_csv("./pathData/path2019-07-16.csv")
    # testpath = data.read_path_csv("./pathData/sample_path.csv")
    testpath = testpath
    near_idx = 0
    try:
        while not rospy.is_shutdown():
            cart_pose = quaternion_to_euler(roscart.selfpose.orientation)
            pos_rad = cart_pose.z
            cartpos = settings.xp.array((roscart.selfpose.position.x, roscart.selfpose.position.y, pos_rad),settings.xp.float32)
            #
            idx = data.get_nearly_point_idx(testpath,cartpos)
            testpath_es, idx_list = data.get_evenly_spaced_points(testpath[idx::],interval)
            if len(testpath_es) < num_step+1:
                print('finished')
                break;
            input_path_global = testpath_es[1:num_step+1]
            input_path_local = coordinate.globalpos_to_localpos(input_path_global,cartpos)
            print('')
            # print('selfpos')
            # print(cartpos)
            # print('input_global')
            # print(input_path_global[:,0:2])
            print('input[x,y]')
            print(input_path_local[:,0:2])
            x = settings.xp.array([input_path_local[:,0:2].flatten()], dtype=settings.xp.float32)
            t_start= time.time()
            y = forward(model, x)
            t_elapsed = time.time() - t_start
            print('time:',t_elapsed,'[sec]')
            print('output[v,w]')
            print(y.data[0])
            params = y.data[0]
            v = params[0,0]
            w = params[0,1]
            '''
            if v < 0:
                v = 0.0
            if w < -settings.xp.pi/2 or settings.xp.pi/2 < w:
                w = max(min(w, settings.xp.pi/2),-settings.xp.pi/2)
            '''
            roscart.move(v,w)
            roscart.set_path_full(testpath[:,0:2])
            path_plan = settings.xp.vstack((cartpos,input_path_global))
            roscart.set_path_input(path_plan[:,0:2])
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
    def set_path_input(self, points):
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
        self.pub_path_input.publish(path)
    def get_position(self, amcl_base):
        self.selfpose = amcl_base.pose.pose
        self.selfpose.position.x = self.selfpose.position.x - self.offset
        self.selfpose.position.y = self.selfpose.position.y - self.offset

if __name__=='__main__':
    settings.set_gpu(-1)
    main()
