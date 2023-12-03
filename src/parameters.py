import numpy as np

SAMPLING_TIME = 0.1
MMS_PER_MOTOR_SPEED = 0.4348
TARGET_RADIUS = 30.0
MARKER_SIZE = 57.0
ROBOT_MARKER_FRONT_LEFT = np.array([-MARKER_SIZE * 0.5, MARKER_SIZE * 0.5])
ROBOT_MARKER_FRONT_RIGHT = np.array([MARKER_SIZE * 0.5, MARKER_SIZE * 0.5])
ROBOT_MARKER_REAR_RIGHT = np.array([MARKER_SIZE * 0.5, -MARKER_SIZE * 0.5])
ROBOT_MARKER_REAR_LEFT = np.array([-MARKER_SIZE * 0.5, -MARKER_SIZE * 0.5])
ROBOT_RADIUS = 80
ROBOT_OUTLINE = np.array([[-56, -30],
                          [56, -30],
                          [56, 55],
                          [34, 72],
                          [0, 80],
                          [-34, 72],
                          [-56, 55]])
