import numpy as np
import typing
import asyncio
from threading import Timer
import time
from tdmclient import ClientAsync

from controller import *
from camera_calibration import *
from global_map import *
from parameters import *
from image_processing import *
from kalman_filter import *

g_is_running = True

prev_x_est = np.zeros((3, 1))
prev_P_est = 1000 * np.ones(3)
prev_input = np.zeros(2)
path_world = []
angle_error = 0
dist_error = 0
switch = 0
sample_time = 0.1


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval) and g_is_running:
            self.function(*self.args, **self.kwargs)


def build_static_graph(img: np.ndarray, dilation_radius_px: int, robot_position: typing.Optional[np.ndarray],
                       robot_radius_px: int, target_position: typing.Optional[np.ndarray], target_radius_px: int,
                       marker_size_px: int, map_width_px: int, map_height_px: int) -> tuple[
    list[list[np.ndarray]], Graph]:
    # Note: the minimum distance to any obstacle is 'dilation_size_px - approx_poly_epsilon'
    approx_poly_epsilon = 2
    obstacle_mask = get_obstacle_mask(img, dilation_radius_px, robot_position, robot_radius_px, target_position,
                                      target_radius_px, marker_size_px, map_width_px, map_height_px)
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


def get_robot_outline(x: float, y: float, theta: float) -> np.ndarray:
    """
    Returns the robot outline in world space
    """
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pos = np.array([x, y])
    return pos + (rot @ ROBOT_OUTLINE.T).T


async def main():
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()

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

    global graph
    global regions
    global path
    global source_position
    global stored_target_position
    global free_source
    global free_target
    global loop_index

    graph = None
    regions = None
    path = []
    source_position = np.zeros(2)
    stored_target_position = np.zeros(2)
    free_source = None
    free_target = None
    loop_index = 0

    dilation_radius_px = int((ROBOT_RADIUS + 10) / map_width_mm * map_width_px)
    robot_radius_px = int((ROBOT_RADIUS + 20) / map_width_mm * map_width_px)
    target_radius_px = int((TARGET_RADIUS + 20) / map_width_mm * map_width_px)
    marker_size_px = int((MARKER_SIZE + 5) / map_width_mm * map_width_px)

    def callback():
        global g_is_running
        global graph
        global regions
        global path
        global path_world
        global source_position
        global stored_target_position
        global free_source
        global free_target
        global prev_x_est
        global prev_P_est
        global prev_input
        global loop_index
        global dist_error
        global angle_error
        global switch
        global sample_time

        if not cap.isOpened():
            g_is_running = False
            return

        ret, frame = cap.read()
        if not ret:
            g_is_running = False
            print('Cannot read frame')
            return

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
            robot_x = robot_position_world.item(0)
            robot_y = robot_position_world.item(1)
            robot_theta = np.arctan2(-robot_direction[0], -robot_direction[1])
            outline = get_robot_outline(robot_x, robot_y, robot_theta).astype(np.int32)
            outline = np.array([transform_affine(world_to_image, pt) for pt in outline], dtype=np.int32)
            cv2.polylines(img_map, [outline], isClosed=True, color=(0, 0, 255), thickness=2,
                          lineType=cv2.LINE_AA)

            measurements = np.array([[robot_x], [robot_y], [robot_theta]])
            if loop_index == 0:
                prev_x_est[:] = measurements
                if len(path_world) != 0:
                    dist_error = np.sqrt(
                        (path_world[0, 0] - measurements[0, 0]) ** 2 + (path_world[0, 1] - measurements[1, 0]) ** 2)
                    angle_error = np.rad2deg(path_world[0, 2] - measurements[2, 0])

            new_x_est, new_P_est = Algorithm_EKF(measurements, prev_x_est, prev_P_est, prev_input)
            prev_x_est = new_x_est
            prev_P_est = new_P_est

            if len(path_world) != 0:
                prev_x_est = prev_x_est.tolist()
                prev_x_est[2] = np.rad2deg(prev_x_est[2])
                goal_state = [path_world[0, 0], path_world[1, 0], np.rad2deg(path_world[2, 0])]
                prev_input, switch, angle_error, dist_error = control(prev_x_est, goal_state, switch, angle_error,
                                                                      dist_error, sample_time)
                prev_x_est = np.array(prev_x_est)
                prev_x_est[2] = np.deg2rad(prev_x_est[2])
                node.send_set_variables(set_robot_speed(prev_input[0], prev_input[1]))

            print(f'x={new_x_est}, Sigma={new_P_est}')

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
            if free_source is not None and free_target is not None:
                draw_path(img_map, graph, path, source_position, free_source, stored_target_position, free_target)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            g_is_running = False
            return
        elif key == ord('m'):
            regions, graph = build_static_graph(frame_map,
                                                dilation_radius_px,
                                                robot_position if robot_found else None,
                                                robot_radius_px,
                                                target_position if target_found else None,
                                                target_radius_px,
                                                marker_size_px,
                                                map_width_px,
                                                map_height_px)
        elif key == ord('u'):
            if robot_found and target_found:
                source_position = robot_position
                stored_target_position = target_position
                free_source, free_target = update_graph(graph, regions, source_position, stored_target_position)
                path = dijkstra(graph.adjacency, Graph.SOURCE, Graph.TARGET)
                path_world = np.empty((len(path) - 1, 3))
                path_world[:, :2] = np.array(
                    [transform_affine(world_to_image, graph.vertices[path[i]]) for i in range(1, len(path))])
                path_world[:, 2] = np.arctan2(path_world[:, 1], path_world[:, 0])

        cv2.imshow('Undistorted frame', img_undistorted)
        cv2.imshow('Map', img_map)

        loop_index += 1

    timer = RepeatTimer(0.1, callback)
    timer.start()

    global g_is_running
    while g_is_running:
        time.sleep(1.0 / 30.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main())
