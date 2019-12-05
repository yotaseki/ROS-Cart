import numpy as np

def init():
    global DATA_SIZE 
    global DATA_NUM_WAYPOINTS 
    global DATA_NUM_STEP 
    global DATA_HZ 
    global DATA_V_STEP 
    global DATA_MAX_V_STEP 
    global DATA_W_STEP 
    global DATA_MAX_W_STEP 
    global DATA_NUM_PREVIOUS_U 
    global DATA_RANGE_TRANSLATE 
    global DATA_RANGE_ROTATE 
    DATA_SIZE = 1000
    DATA_NUM_WAYPOINTS = 10
    DATA_NUM_STEP = DATA_NUM_WAYPOINTS
    DATA_HZ = 10
    DATA_V_STEP = 1.0 / DATA_HZ # [m/step]
    DATA_MAX_V_STEP = 1.0 / DATA_HZ # [m/step]
    DATA_W_STEP = (np.pi/6) / DATA_HZ # [rad/step]
    DATA_MAX_W_STEP = (np.pi/6) / DATA_HZ # [rad/step]
    DATA_NUM_PREVIOUS_U = 0
    DATA_RANGE_TRANSLATE = 0
    DATA_RANGE_ROTATE = 0

def set(filename):
    pass
