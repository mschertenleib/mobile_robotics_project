import numpy as np

# Frame
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# Map
MAP_WIDTH_MM = 972
MAP_HEIGHT_MM = 671
MAP_WIDTH_PX = 900
MAP_HEIGHT_PX = int(MAP_WIDTH_PX * MAP_HEIGHT_MM / MAP_WIDTH_MM)

# Control
SAMPLING_TIME = 0.1

# Physical parameters
MMS_PER_MOTOR_SPEED = 0.4348
TARGET_RADIUS_MM = 70.0
MARKER_SIZE_MM = 57.0
ROBOT_MARKER_FRONT_LEFT = np.array([-MARKER_SIZE_MM * 0.5, MARKER_SIZE_MM * 0.5])
ROBOT_MARKER_FRONT_RIGHT = np.array([MARKER_SIZE_MM * 0.5, MARKER_SIZE_MM * 0.5])
ROBOT_MARKER_REAR_RIGHT = np.array([MARKER_SIZE_MM * 0.5, -MARKER_SIZE_MM * 0.5])
ROBOT_MARKER_REAR_LEFT = np.array([-MARKER_SIZE_MM * 0.5, -MARKER_SIZE_MM * 0.5])
ROBOT_RADIUS_MM = 80
ROBOT_CENTER_TO_WHEEL = 48
ROBOT_WHEEL_RADIUS = 22
ROBOT_OUTLINE = np.array([[-56, -30],
                          [56, -30],
                          [56, 55],
                          [34, 72],
                          [0, 80],
                          [-34, 72],
                          [-56, 55]])

# Dimension parameters in pixels
DILATION_RADIUS_PX = int((ROBOT_RADIUS_MM + 10) / MAP_WIDTH_MM * MAP_WIDTH_PX)
ROBOT_RADIUS_PX = int((ROBOT_RADIUS_MM + 20) / MAP_WIDTH_MM * MAP_WIDTH_PX)
TARGET_RADIUS_PX = int((TARGET_RADIUS_MM + 10) / MAP_WIDTH_MM * MAP_WIDTH_PX)
MARKER_SIZE_PX = int((MARKER_SIZE_MM + 5) / MAP_WIDTH_MM * MAP_WIDTH_PX)
