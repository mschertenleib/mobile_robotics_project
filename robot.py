import numpy as np


class Thymio:
    def __init__(self):
        self._pos_x = 0
        self._pos_y = 0
        self._theta = 0
        self._outline = np.array([[-5.5, -5.5],
                                  [5.5, -5.5],
                                  [5.5, 3.0],
                                  [4.7, 3.8],
                                  [2.5, 4.9],
                                  [0.0, 5.5],
                                  [-2.4, 5.0],
                                  [-4.6, 3.9],
                                  [-5.5, 3.0],
                                  [-5.5, -5.5]])
        self._sensor_angles = np.array([120, 105, 90, 75, 60, -90, -90]) * np.pi / 180
        self._sensor_pos = np.array([[-4.6, 3.9],
                                     [-2.4, 5.0],
                                     [0.0, 5.5],
                                     [2.5, 4.9],
                                     [4.7, 3.8],
                                     [3.0, -5.5],
                                     [-3.0, -5.5]])
