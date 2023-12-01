import numpy as np

from camera_calibration import *
from global_map import *


def build_static_graph(img: np.ndarray) -> tuple[list[list[np.ndarray]], Graph]:
    # Note: the minimum distance to any obstacle is 'kernel_size - approx_poly_epsilon'
    approx_poly_epsilon = 2
    obstacle_mask = get_obstacle_mask(img)
    regions = extract_contours(obstacle_mask, approx_poly_epsilon)
    graph = build_graph(regions)
    return regions, graph


def draw_static_graph(img: np.ndarray, graph: Graph, regions: list[list[np.ndarray]]) -> np.ndarray:
    free_space = np.empty_like(img)
    free_space[:] = (64, 64, 192)
    all_contours = [contour for region in regions for contour in region]
    cv2.drawContours(free_space, all_contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)
    graph_img = cv2.addWeighted(img, 0.5, free_space, 0.5, 0.0)
    cv2.drawContours(graph_img, all_contours, contourIdx=-1, color=(64, 64, 192))
    draw_graph(graph_img, graph)
    return graph_img


def build_draw_dynamic_graph(img: np.ndarray, graph: Graph, regions: list[list[np.ndarray]], source: np.ndarray,
                             target: np.ndarray) -> np.ndarray:
    free_source, free_target = update_graph(graph, regions, source, target)
    path = dijkstra(graph.adjacency, Graph.SOURCE, Graph.TARGET)
    graph_img = img.copy()
    draw_path(graph_img, graph, path, source, free_source, target, free_target)
    return graph_img


def main():
    width_px = 841
    height_px = 594
    warped = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    warped_img = warped.copy()

    width_mm = 1189 - 25
    height_mm = 841 - 25
    image_to_world = get_image_to_world(width_px, height_px, width_mm, height_mm)
    world_to_image = get_world_to_image(width_mm, height_mm, width_px, height_px)

    thymio = Thymio()

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    frame_width = 1280
    frame_height = 720
    # camera_matrix, distortion_coeffs = calibrate_camera(frame_width, frame_height)
    # store_to_json('camera.json', camera_matrix, distortion_coeffs)
    # return
    camera_matrix, distortion_coeffs = load_from_json('camera.json')
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (width_px, height_px), 0,
                                                           (width_px, height_px))

    graph = None
    regions = None
    static_graph_img = None

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

            matrix = get_perspective_transform(map_corners, width_px, height_px)
            warped = cv2.warpPerspective(undistorted_frame, matrix, (width_px, height_px))
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
            tip = (robot_position + robot_direction * 40).astype(np.int32)
            cv2.arrowedLine(warped_img, pos, tip, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA, tipLength=0.5)
            cv2.drawMarker(warped_img, position=pos, color=(0, 0, 255), thickness=2, markerSize=10,
                           markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)

            position_world = transform_affine(image_to_world, robot_position)
            print(position_world)
            thymio.pos_x = position_world[0]
            thymio.pos_y = position_world[1]
            thymio.theta = np.arctan2(-robot_direction[0], -robot_direction[1])
            outline = thymio.get_outline().astype(np.int32)
            outline = np.array([transform_affine(world_to_image, pt) for pt in outline], dtype=np.int32)
            cv2.polylines(warped_img, [outline], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

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

        if static_graph_img is not None and graph is not None and robot_found and target_found:
            graph_img = build_draw_dynamic_graph(static_graph_img, graph, regions, robot_position, target_position)
            cv2.imshow('graph', graph_img)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord('m'):
            regions, graph = build_static_graph(warped)
            static_graph_img = draw_static_graph(warped, graph, regions)
            cv2.imshow('graph', static_graph_img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
