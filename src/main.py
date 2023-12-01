import numpy as np

from camera_calibration import *
from global_map import *


def build_static_graph(img: np.ndarray, dilation_size_px: int) -> tuple[list[list[np.ndarray]], Graph]:
    # Note: the minimum distance to any obstacle is 'dilation_size_px - approx_poly_epsilon'
    approx_poly_epsilon = 2
    obstacle_mask = get_obstacle_mask(img, dilation_size_px)
    regions = extract_contours(obstacle_mask, approx_poly_epsilon)
    graph = build_graph(regions)
    return regions, graph


def draw_static_graph(img: np.ndarray, graph: Graph, regions: list[list[np.ndarray]]):
    free_space = np.empty_like(img)
    free_space[:] = (64, 64, 192)
    all_contours = [contour for region in regions for contour in region]
    cv2.drawContours(free_space, all_contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)
    cv2.addWeighted(img, 0.5, free_space, 0.5, 0.0, dst=img)
    cv2.drawContours(img, all_contours, contourIdx=-1, color=(64, 64, 192))
    draw_graph(img, graph)


def build_draw_dynamic_graph(img: np.ndarray, graph: Graph, regions: list[list[np.ndarray]], source: np.ndarray,
                             target: np.ndarray):
    free_source, free_target = update_graph(graph, regions, source, target)
    path = dijkstra(graph.adjacency, Graph.SOURCE, Graph.TARGET)
    draw_path(img, graph, path, source, free_source, target, free_target)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_width = 960
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    map_width_mm = 1000 - 25
    map_height_mm = 696 - 25
    map_width_px = 900
    map_height_px = int(map_width_px * map_height_mm / map_width_mm)
    image_to_world = get_image_to_world_matrix(map_width_px, map_height_px, map_width_mm, map_height_mm)
    world_to_image = get_world_to_image_matrix(map_width_mm, map_height_mm, map_width_px, map_height_px)

    # Convention: in main(), all frame_* images are destined to image processing, and have no extra drawings on them.
    # All img_* images are the ones which have extra text, markers, etc. drawn on them.

    # Undistorted
    frame_undistorted = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    img_undistorted = np.zeros_like(frame_undistorted)

    # Undistorted, perspective corrected
    frame_map = np.zeros((map_height_px, map_width_px, 3), dtype=np.uint8)
    img_map = np.zeros_like(frame_map)
    base_img_map = np.zeros_like(frame_map)

    thymio = Thymio()

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # camera_matrix, distortion_coeffs = calibrate_camera(frame_width, frame_height)
    # store_to_json('camera.json', camera_matrix, distortion_coeffs)
    # return
    camera_matrix, distortion_coeffs = load_from_json('camera.json')
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs,
                                                           (map_width_px, map_height_px), 0,
                                                           (map_width_px, map_height_px))

    graph = None
    regions = None

    dilation_size_mm = Thymio.RADIUS + 20
    dilation_size_px = int(dilation_size_mm / map_width_mm * map_width_px)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        cv2.undistort(frame, camera_matrix, distortion_coeffs, dst=frame_undistorted, newCameraMatrix=new_camera_matrix)
        img_undistorted[:] = frame_undistorted

        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        map_found, map_corners = detect_map(corners, ids)
        text_y = 30
        if not map_found:
            cv2.putText(img_undistorted, 'Map not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 30
            img_map[:] = base_img_map
        else:
            cv2.putText(img_undistorted, 'Map detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 30
            for corner in map_corners:
                cv2.drawMarker(img_undistorted, position=corner.astype(np.int32), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, thickness=2)

            matrix = get_perspective_transform(map_corners, map_width_px, map_height_px)
            cv2.warpPerspective(frame_undistorted, matrix, dsize=(map_width_px, map_height_px), dst=frame_map)
            base_img_map[:] = frame_map
            img_map[:] = frame_map

        corners, ids, rejected = detector.detectMarkers(frame_map)

        robot_found, robot_position, robot_direction = detect_robot(corners, ids)
        target_found, target_position = detect_target(corners, ids)

        text_y = 30
        if not robot_found:
            cv2.putText(img_map, 'Robot not detected', org=(10, text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 30
        else:
            cv2.putText(img_map, 'Robot detected', org=(10, text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 30

            pos = robot_position.astype(np.int32)
            tip = (robot_position + robot_direction).astype(np.int32)
            cv2.arrowedLine(img_map, pos, tip, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA,
                            tipLength=0.5)
            cv2.drawMarker(img_map, position=pos, color=(0, 0, 255), thickness=2, markerSize=10,
                           markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)

            robot_position_world = transform_affine(image_to_world, robot_position)
            thymio.pos_x = robot_position_world[0]
            thymio.pos_y = robot_position_world[1]
            thymio.theta = np.arctan2(-robot_direction[0], -robot_direction[1])
            outline = thymio.get_outline().astype(np.int32)
            outline = np.array([transform_affine(world_to_image, pt) for pt in outline], dtype=np.int32)
            cv2.polylines(img_map, [outline], isClosed=True, color=(0, 0, 255), thickness=2,
                          lineType=cv2.LINE_AA)

        if not target_found:
            cv2.putText(img_map, 'Target not detected', org=(10, text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 30
        else:
            cv2.putText(img_map, 'Target detected', org=(10, text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 30
            cv2.drawMarker(img_map, position=target_position.astype(np.int32), color=(0, 255, 0),
                           markerSize=10,
                           markerType=cv2.MARKER_CROSS)

        if graph is not None:
            draw_static_graph(img_map, graph, regions)
            if robot_found and target_found:
                build_draw_dynamic_graph(img_map, graph, regions, robot_position, target_position)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord('m'):
            regions, graph = build_static_graph(frame_map, dilation_size_px)

        cv2.imshow('Undistorted frame', img_undistorted)
        cv2.imshow('Map', img_map)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
