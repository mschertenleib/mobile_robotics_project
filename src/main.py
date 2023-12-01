from camera_calibration import *
from global_map import *


def build_draw_graph(img):
    # Note: the minimum distance to any obstacle is 'kernel_size - approx_poly_epsilon'
    approx_poly_epsilon = 2
    obstacle_mask = get_obstacle_mask(img)
    regions = extract_contours(obstacle_mask, approx_poly_epsilon)

    free_space = np.empty_like(img)
    free_space[:] = (64, 64, 192)
    all_contours = [contour for region in regions for contour in region]
    cv2.drawContours(free_space, all_contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)
    img = cv2.addWeighted(img, 0.5, free_space, 0.5, 0.0)
    cv2.drawContours(img, all_contours, contourIdx=-1, color=(64, 64, 192))

    graph = build_graph(regions)
    draw_graph(img, graph)

    return img


def main():
    dst_width = 841
    dst_height = 594
    warped = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    warped_img = warped.copy()

    width_mm = 1189
    height_mm = 841
    image_to_world = get_image_to_world(dst_width, dst_height, width_mm, height_mm)
    world_to_image = get_world_to_image(width_mm, height_mm, dst_width, dst_height)

    thymio = Thymio()

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    frame_width = 1280
    frame_height = 720
    # camera_matrix, distortion_coeffs = calibrate_camera(frame_width, frame_height)
    # store_to_json('camera.json', camera_matrix, distortion_coeffs)
    camera_matrix, distortion_coeffs = load_from_json('camera.json')
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (dst_width, dst_height), 0,
                                                           (dst_width, dst_height))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        text_y = 25

        undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coeffs, None, new_camera_matrix)
        frame_img = undistorted_frame.copy()

        corners, ids, rejected = detector.detectMarkers(undistorted_frame)

        map_found, map_corners = detect_map(corners, ids)
        if not map_found:
            cv2.putText(frame_img, 'Map not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(frame_img, 'Map detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20
            for corner in map_corners:
                cv2.drawMarker(frame_img, position=corner.astype(np.int32), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, thickness=2)

            matrix = get_perspective_transform(map_corners, dst_width, dst_height)
            warped = cv2.warpPerspective(undistorted_frame, matrix, (dst_width, dst_height))
            warped_img = warped.copy()

        cv2.imshow('frame_img', frame_img)

        corners, ids, rejected = detector.detectMarkers(warped_img)

        robot_found, robot_position, robot_direction = detect_robot(corners, ids)
        target_found, target_position = detect_target(corners, ids)

        text_y = 25
        if not robot_found:
            cv2.putText(warped_img, 'Robot not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(warped_img, 'Robot detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20

            pos = robot_position.astype(np.int32)
            tip = (robot_position + robot_direction).astype(np.int32)
            cv2.line(warped_img, pos, tip, color=(0, 0, 255))
            cv2.drawMarker(warped_img, position=pos, color=(0, 0, 255), markerSize=10,
                           markerType=cv2.MARKER_CROSS)

            # FIXME: this is just a temporary hack
            # distance_center_back_px = np.abs(thymio.POINT_BACK[1]) / width_mm * dst_width
            # position_img, direction_img = get_robot_pose_old(robot_vertices, distance_center_back_px)
            # cv2.circle(warped_img, center=position_img.astype(np.int32), radius=4, color=(0, 255, 0), thickness=-1,
            #           lineType=cv2.LINE_AA)
            # cv2.arrowedLine(warped_img, position_img.astype(np.int32),
            #                (position_img + direction_img * 40).astype(np.int32),
            #                color=(0, 255, 0), thickness=2, line_type=cv2.LINE_AA, tipLength=0.5)
            # position_world = transform_affine(image_to_world, position_img)
            # thymio.pos_x = position_world[0]
            # thymio.pos_y = position_world[1]
            # thymio.theta = np.arctan2(-direction_img[0], -direction_img[1])
            # outline = thymio.get_outline().astype(np.int32)
            # outline = np.array([transform_affine(world_to_image, pt) for pt in outline], dtype=np.int32)
            # cv2.polylines(warped_img, [outline], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        if not target_found:
            cv2.putText(warped_img, 'Target not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(warped_img, 'Target detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20
            cv2.drawMarker(warped_img, position=target_position.astype(np.int32), color=(0, 255, 0), markerSize=10,
                           markerType=cv2.MARKER_CROSS)

        cv2.imshow('warped_img', warped_img)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord('m'):
            graph_img = build_draw_graph(warped)
            cv2.imshow('graph', graph_img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
