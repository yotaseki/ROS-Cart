import rospy
from ros_tools import Controller,Navigator
import xp_settings as settings
import numpy as np
import matplotlib.pyplot as plt
settings.set_gpu(-1)
xp = settings.xp

rospy.init_node('CartController',anonymous=True)
rate = rospy.Rate(10)
c = Controller()
n = Navigator(10,0.05)
v = 0.5
w = np.pi / 2
path = []
path_odom = []
for t in range(100):
    c.command_vel(v,w)
    rate.sleep()
    selfpos_odom = n.get_position3D(n.selfpose)
    selfpos = n.get_position3D(n.selfpose_t)
    path.append(selfpos)
    path_odom.append(selfpos_odom)
    print('Odometry   :',selfpos_odom)
    print('Modelstate :',selfpos)
path = np.array(path)
path_odom = np.array(path_odom)
plt.plot(path[:,0], path[:,1])
plt.plot(path_odom[:,0], path_odom[:,1])
plt.show()
