from robot import *

import cv2
import numpy as np


def transform(matrix, point):
    transformed = matrix @ np.float32([point[0], point[1], 1])
    return transformed[0:2] / transformed[2]


def correct_perspective():
    img = cv2.imread('perspective_box.jpg')
    assert img is not None
    img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))

    pts_src = np.float32([[226, 173], [408, 273], [78, 275], [258, 424]])
    dst_width = 146 * 4
    dst_height = 126 * 4
    pts_dst = np.float32([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    inv_matrix = np.linalg.inv(matrix)

    for pt in pts_src:
        cv2.circle(img, center=pt.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    center = transform(inv_matrix, [dst_width / 3, dst_height / 3]).astype(int)
    cv2.circle(img, center=center, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.imshow('main', img)
    cv2.waitKey(0)

    img = cv2.warpPerspective(img, matrix, (dst_width, dst_height))
    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reconstruct_thymio():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # TODO: convert to HSV for detecting dots? But might actually be worse
    tip = np.float32([200, 140])
    base = np.float32([50, 300])
    dir = tip - base
    dir /= np.linalg.norm(dir)
    left = base - np.array([-dir[1], dir[0]]) * 50
    right = base + np.array([-dir[1], dir[0]]) * 50
    points = np.array([tip, left, right], dtype=np.int32)
    # TODO: it might actually be better to simply reconstruct the middle point from the left and right (we can detect
    #  which one is the tip because it is the furthest away from all others), and directly use that to reconstruct
    #  the position and orientation, without the need to do a line fit
    [vx, vy, x, y] = cv2.fitLine(points, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
    pt1 = (np.array([x, y]).squeeze() - np.array([vx, vy]).squeeze() * 600).astype(int)
    pt2 = (np.array([x, y]).squeeze() + np.array([vx, vy]).squeeze() * 600).astype(int)
    cv2.line(img, pt1, pt2, color=(255, 255, 255))
    cv2.circle(img, center=tip.astype(int), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(img, center=base.astype(int), radius=5, color=(0, 255, 255), thickness=-1)
    cv2.circle(img, center=left.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, center=right.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # correct_perspective()
    reconstruct_thymio()
