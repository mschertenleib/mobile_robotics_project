import numpy as np


class Thymio:
    # X axis towards the right
    # Y axis towards the front
    # theta from x to y
    # All distances are in mm, angles in radians

    POINT_BACK = np.array([0, -16])
    POINT_FRONT_LEFT = np.array([-35, 56])
    POINT_FRONT_RIGHT = np.array([35, 56])
    RADIUS = 80
    _OUTLINE = np.array([[-56, -30],
                         [56, -30],
                         [56, 55],
                         [34, 72],
                         [0, 80],
                         [-34, 72],
                         [-56, 55]])

    def __init__(self, pos_x: float = 0.0, pos_y: float = 0.0, theta: float = 0.0):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta

    def get_outline(self):
        rot = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        pos = np.array([self.pos_x, self.pos_y])
        return pos + (rot @ self._OUTLINE.T).T
