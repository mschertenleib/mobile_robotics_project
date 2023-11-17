from robot import *

import cv2 as cv
import numpy as np


def transform(matrix, point):
    transformed = matrix @ np.float32([point[0], point[1], 1])
    return transformed[0:2] / transformed[2]


def draw_thymio(img, position: np.ndarray, angle: float):
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = position.astype(float) + (rot @ Thymio.outline.T).T
    cv.polylines(img, [points.astype(int)], isClosed=True, color=(255, 255, 255))


def correct_perspective():
    img = cv.imread('perspective_box.jpg')
    assert img is not None
    img = cv.resize(img, (img.shape[1] // 8, img.shape[0] // 8))

    pts_src = np.float32([[226, 173], [408, 273], [78, 275], [258, 424]])
    dst_width = 146 * 4
    dst_height = 126 * 4
    pts_dst = np.float32([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]])
    matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
    inv_matrix = np.linalg.inv(matrix)

    for pt in pts_src:
        cv.circle(img, center=pt.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    center = transform(inv_matrix, [dst_width / 3, dst_height / 3]).astype(int)
    cv.circle(img, center=center, radius=5, color=(255, 0, 0), thickness=-1)
    cv.imshow('main', img)
    cv.waitKey(0)

    img = cv.warpPerspective(img, matrix, (dst_width, dst_height))
    cv.imshow('main', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def reconstruct_thymio():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # TODO: convert to HSV for detecting dots? But might actually be worse

    tip = np.float32([200, 140])
    left = np.float32([180, 280])
    right = np.float32([300, 250])
    center = (left + right) * 0.5

    cv.line(img, center.astype(int), tip.astype(int), color=(255, 255, 255))
    cv.circle(img, center=tip.astype(int), radius=5, color=(255, 0, 0), thickness=-1)
    cv.circle(img, center=left.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    cv.circle(img, center=right.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
    cv.circle(img, center=center.astype(int), radius=5, color=(255, 255, 255), thickness=-1)

    dir = tip - center
    draw_thymio(img, center, 0)  # TODO angle

    cv.imshow('main', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # correct_perspective()
    reconstruct_thymio()
