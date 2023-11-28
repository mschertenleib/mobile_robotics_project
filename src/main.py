from global_map import *


def build_draw_graph(img):
    # Note: the minimum distance to any obstacle is 'kernel_size - approx_poly_epsilon'
    approx_poly_epsilon = 2
    obstacle_mask = get_obstacle_mask(img)
    regions = extract_contours(obstacle_mask, approx_poly_epsilon)
    all_contours = [contour for region in regions for contour in region]

    free_space = np.empty_like(img)
    free_space[:] = (64, 64, 192)
    cv2.drawContours(free_space, all_contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)
    img = cv2.addWeighted(img, 0.5, free_space, 0.5, 0.0)
    cv2.drawContours(img, all_contours, contourIdx=-1, color=(64, 64, 192))

    graph = build_graph(all_contours)
    draw_graph(img, graph)

    return img


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    dst_width = 594
    dst_height = 841
    warped = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    warped_img = warped.copy()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        filtered_frame = cv2.bilateralFilter(frame, 15, 150, 150)
        hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)

        text_y = 25
        frame_img = frame.copy()
        map_vertices = detect_map_corners(hsv)
        if map_vertices is None:
            cv2.putText(frame_img, 'Map not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(frame_img, 'Map detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20
            for vertex in map_vertices:
                cv2.drawMarker(frame_img, position=vertex, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)

            matrix = get_perspective_transform(map_vertices, dst_width, dst_height)
            warped = cv2.warpPerspective(filtered_frame, matrix, (dst_width, dst_height))
            warped_img = warped.copy()

        cv2.imshow('frame_img', frame_img)

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        robot_vertices = detect_robot_vertices(hsv)
        text_y = 25
        if robot_vertices is None:
            cv2.putText(warped_img, 'Robot not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(warped_img, 'Robot detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20
            for vertex in robot_vertices:
                cv2.drawMarker(warped_img, position=vertex, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)

            distance_center_back = 10  # FIXME
            position, direction = get_robot_pose(robot_vertices, distance_center_back)
            cv2.circle(warped_img, center=position.astype(np.int32), radius=4, color=(0, 255, 0), thickness=-1,
                       lineType=cv2.LINE_AA)
            cv2.arrowedLine(warped_img, position.astype(np.int32), (position + direction + 10).astype(np.int32),
                            color=(0, 255, 0), thickness=2, line_type=cv2.LINE_AA, tipLength=0.5)

        target = detect_target(hsv)
        if target is None:
            cv2.putText(warped_img, 'Target not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(warped_img, 'Target detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20
            cv2.drawMarker(warped_img, position=target, color=(255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=2)

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
