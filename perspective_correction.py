import cv2
import numpy as np


def main():
    img = cv2.imread('perspective_box.jpg')
    assert img is not None
    img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))

    pts_src = np.float32([[226, 173], [408, 273], [78, 275], [258, 424]])
    for pt in pts_src:
        cv2.circle(img, center=pt.astype(int), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow('main', img)
    cv2.waitKey(0)

    dst_width = 146 * 4
    dst_height = 126 * 4
    pts_dst = np.float32([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    img = cv2.warpPerspective(img, matrix, (dst_width, dst_height))

    cv2.imshow('main', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
