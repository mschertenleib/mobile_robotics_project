import numpy as np


class Thymio:
    # All distances are in mm, angles in radians
    outline = np.array([[-55, -55],
                        [55, -55],
                        [55, 30],
                        [47, 38],
                        [25, 49],
                        [0, 55],
                        [-24, 50],
                        [-46, 39],
                        [-55, 30],
                        [-55, -55]])
    sensor_angles = np.radians(np.array([120, 105, 90, 75, 60, -90, -90]))
    sensor_pos = np.array([[-46, 39],
                           [-24, 50],
                           [0, 55],
                           [25, 49],
                           [47, 38],
                           [30, -55],
                           [-30, -55]])

    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0
        self.theta = 0
