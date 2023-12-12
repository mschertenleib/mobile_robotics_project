import cv2

from image_processing import *

g_mouse_x, g_mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global g_mouse_x, g_mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        g_mouse_x, g_mouse_y = x, y


def inspect_image():
    img = cv2.imread('../images/camera_frame.png')

    global g_mouse_x, g_mouse_y

    while True:
        bgr = img[g_mouse_y, g_mouse_x]
        print(f'XY = ({g_mouse_x}, {g_mouse_y}), RGB = {bgr[::-1]}')

        cv2.imshow('main', img)
        cv2.setMouseCallback('main', mouse_callback)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cv2.destroyAllWindows()


def main():
    frame = cv2.imread('../images/camera_frame.png')

    map_corners = np.array([[11, 17], [823, 28], [843, 612], [18, 632]])

    matrix = get_perspective_transform(map_corners, MAP_WIDTH_PX, MAP_HEIGHT_PX)
    warped = cv2.warpPerspective(frame, matrix, dsize=(MAP_WIDTH_PX, MAP_HEIGHT_PX))
    cv2.imwrite('../images/obstacle_extraction/warped.png', warped)

    img = warped.copy()

    threshold = 120
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)
    cv2.imwrite('../images/obstacle_extraction/thresholded.png', img * 255)

    robot_position = np.array([181, 456])
    target_position = np.array([483, 336])

    cv2.circle(img, center=robot_position.astype(np.int32), radius=ROBOT_MASK_RADIUS_PX, color=[0], thickness=-1)
    cv2.circle(img, center=target_position.astype(np.int32), radius=TARGET_MASK_RADIUS_PX, color=[0], thickness=-1)
    cv2.rectangle(img, pt1=(0, 0), pt2=(MARKER_MASK_SIZE_PX, MARKER_MASK_SIZE_PX), color=[0], thickness=-1)
    cv2.rectangle(img, pt1=(MAP_WIDTH_PX - MARKER_MASK_SIZE_PX, 0), pt2=(MAP_WIDTH_PX, MARKER_MASK_SIZE_PX),
                  color=[0],
                  thickness=-1)
    cv2.rectangle(img, pt1=(0, MAP_HEIGHT_PX - MARKER_MASK_SIZE_PX), pt2=(MARKER_MASK_SIZE_PX, MAP_HEIGHT_PX),
                  color=[0],
                  thickness=-1)
    cv2.rectangle(img, pt1=(MAP_WIDTH_PX - MARKER_MASK_SIZE_PX, MAP_HEIGHT_PX - MARKER_MASK_SIZE_PX),
                  pt2=(MAP_WIDTH_PX, MAP_HEIGHT_PX), color=[0], thickness=-1)
    cv2.imwrite('../images/obstacle_extraction/masked.png', img * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, dst=img, iterations=1)
    cv2.imwrite('../images/obstacle_extraction/opened.png', img * 255)

    img[:, 0] = 1
    img[:, -1] = 1
    img[0, :] = 1
    img[-1, :] = 1
    cv2.imwrite('../images/obstacle_extraction/borders.png', img * 255)

    dilation_kernel_size = 2 * DILATION_RADIUS_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    img = cv2.dilate(img, kernel)
    cv2.imwrite('../images/obstacle_extraction/dilated.png', img * 255)

    img = 1 - img
    cv2.imwrite('../images/obstacle_extraction/final.png', img * 255)


if __name__ == '__main__':
    # inspect_image()
    main()
