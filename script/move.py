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
import time

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

def translation(p, x, y):
    pass

def rotation(p, rad):
    pass

def global2local(path, pos):
    pos_x = pos[0]
    pos_y = pos[1]
    pos_rad = - pos[2]
    X = path[:,0]
    Y = path[:,1]
    X = (X * np.cos(pos_rad)) - (Y * np.sin(pos_rad)) - pos_x
    Y = (X * np.sin(pos_rad)) + (Y * np.cos(pos_rad)) - pos_y
    R = path[:,2] - pos_rad
    ret = np.vstack((X,Y,R))
    ret = np.transpose(ret)
    return ret

def main():
    gpu_idx = -1
    if(gpu_idx >= 0):
        xp = cp
    else:
        xp = np
    # ROS
    roscart = ROSCart()
    rate = rospy.Rate(10); # 10Hz / 0.1sec
    # MODEL
    num_step = 9
    input_dim = num_step*2
    model = Generator(input_dim, num_step)
    # LOAD WEIGHT
    weight = sys.argv[1]
    serializers.load_npz(weight, model)
    # TEST PATH
    space = 100.0 / 1000.0
    # points = xp.array([[1.0,0.0],[2.0,0.0],[3.0,0.0]])
    pdata = PathData(gpu_idx)
    k =  xp.pi / 18   # curvature
    testpath = pdata.make_arc_path_2(10, k)
    near_idx = 0
    t_start= time.time()
    try:
        while not rospy.is_shutdown():
            cart_pose = quaternion_to_euler(roscart.selfpose.orientation)
            pos_rad = cart_pose.z
            cartpos = np.array((roscart.selfpose.position.x, roscart.selfpose.position.y, pos_rad),np.float32)
            #
            idx = pdata.get_nearly_point_idx(testpath,cartpos)
            testpath_es, idx_list = pdata.get_evenly_spaced_points(testpath[idx::],space)
            if len(testpath_es) < num_step+1:
                print('finished')
                break;
            input_path_global = testpath_es[1:num_step+1]
            input_path_local = global2local(input_path_global,cartpos)
            print('')
            # print('selfpos')
            # print(cartpos)
            # print('input_global')
            # print(input_path_global[:,0:2])
            print('input[x,y]')
            print(input_path_local[:,0:2])
            x = xp.array([input_path_local[:,0:2].flatten()], dtype=np.float32)
            y = forward(model, x)
            print('output[v,w]')
            print(y.data[0])
            params = y.data[0]
            v = params[0,0]
            w = params[0,1]
            '''
            if v < 0:
                v = 0.0
            if w < -np.pi/2 or np.pi/2 < w:
                w = max(min(w, np.pi/2),-np.pi/2)
            '''
            roscart.move(v,w)
            roscart.set_path_full(testpath[:,0:2])
            path_plan = np.vstack((cartpos,input_path_global))
            roscart.set_path_input(path_plan[:,0:2])
            rate.sleep()
            t_elapsed = time.time() - t_start
            t_start= time.time()
            print('time:',t_elapsed,'[sec]')
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
    main()
