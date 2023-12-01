import numpy as np

from camera_calibration import *
from global_map import *


def build_static_graph(img: np.ndarray) -> tuple[list[list[np.ndarray]], Graph]:
    # Note: the minimum distance to any obstacle is 'kernel_size - approx_poly_epsilon'
    approx_poly_epsilon = 2
    dilation_size_px = 50
    obstacle_mask = get_obstacle_mask(img, dilation_size_px)
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

    # Convention: in main(), all frame_* images are destined to image processing, and have no extra drawings on them.
    # All img_* images are the ones which have extra text, markers, etc. drawn on them.

    frame_warped = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    img_warped = frame_warped.copy()

    width_mm = 1000 - 25
    height_mm = 696 - 25
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
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (width_px, height_px), 1,
                                                           (width_px, height_px))

    graph = None
    regions = None
    img_static_graph = None

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame')
            break

        text_y = 25

        frame_undistorted = cv2.undistort(frame, camera_matrix, distortion_coeffs, None, new_camera_matrix)
        img_frame = frame_undistorted.copy()

        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        map_found, map_corners = detect_map(corners, ids)
        if not map_found:
            cv2.putText(img_frame, 'Map not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(img_frame, 'Map detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20
            for corner in map_corners:
                cv2.drawMarker(img_frame, position=corner.astype(np.int32), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, thickness=2)

            matrix = get_perspective_transform(map_corners, width_px, height_px)
            frame_warped = cv2.warpPerspective(frame_undistorted, matrix, (width_px, height_px))
            img_warped = frame_warped.copy()

        cv2.imshow('frame_img', img_frame)

        corners, ids, rejected = detector.detectMarkers(img_warped)

        robot_found, robot_position, robot_direction = detect_robot(corners, ids)
        target_found, target_position = detect_target(corners, ids)

        text_y = 25
        if not robot_found:
            cv2.putText(img_warped, 'Robot not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(img_warped, 'Robot detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20

            pos = robot_position.astype(np.int32)
            tip = (robot_position + robot_direction).astype(np.int32)
            cv2.arrowedLine(img_warped, pos, tip, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA, tipLength=0.5)
            cv2.drawMarker(img_warped, position=pos, color=(0, 0, 255), thickness=2, markerSize=10,
                           markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)

            robot_position_world = transform_affine(image_to_world, robot_position)
            thymio.pos_x = robot_position_world[0]
            thymio.pos_y = robot_position_world[1]
            thymio.theta = np.arctan2(-robot_direction[0], -robot_direction[1])
            outline = thymio.get_outline().astype(np.int32)
            outline = np.array([transform_affine(world_to_image, pt) for pt in outline], dtype=np.int32)
            cv2.polylines(img_warped, [outline], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        if not target_found:
            cv2.putText(img_warped, 'Target not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 20
        else:
            cv2.putText(img_warped, 'Target detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 20
            cv2.drawMarker(img_warped, position=target_position.astype(np.int32), color=(0, 255, 0), markerSize=10,
                           markerType=cv2.MARKER_CROSS)

        cv2.imshow('warped_img', img_warped)

        if img_static_graph is not None and graph is not None and robot_found and target_found:
            img_graph = build_draw_dynamic_graph(img_static_graph, graph, regions, robot_position, target_position)
            cv2.imshow('graph', img_graph)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord('m'):
            regions, graph = build_static_graph(frame_warped)
            img_static_graph = draw_static_graph(frame_warped, graph, regions)
            cv2.imshow('graph', img_static_graph)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
