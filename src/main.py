import asyncio
import time
from threading import Timer

from tdmclient import ClientAsync

from camera_calibration import *
from controller import *
from global_map import *
from image_processing import *
from kalman_filter import *
from parameters import *


class Navigator:
    def __init__(self):
        self.cap = None
        self.node = None
        self.is_running = True
        self.loop_index = 0
        # Image
        self.camera_matrix = None
        self.new_camera_matrix = None
        self.roi = None
        self.distortion_coeffs = None
        self.frame_undistorted = None
        self.img_undistorted = None
        self.frame_map = None
        self.img_map = None
        self.base_img_map = None
        self.detector = None
        self.image_to_world = None
        self.world_to_image = None
        self.map_width_px = 0
        self.map_height_px = 0
        self.dilation_radius_px = 0
        self.robot_radius_px = 0
        self.target_radius_px = 0
        self.marker_size_px = 0
        # Map
        self.regions = None
        self.graph = None
        self.path: list[int] = []
        self.path_world = None
        self.source_position = np.zeros(2)
        self.free_source = np.zeros(2)
        self.stored_target_position = np.zeros(2)
        self.free_target = np.zeros(2)
        # Kalman filter
        self.prev_x_est = np.zeros((3, 1))
        self.prev_P_est = 1000 * np.ones(3)
        self.prev_input = np.zeros(2)
        # Controller
        self.angle_error = 0.0
        self.dist_error = 0.0
        self.switch = 0
        self.sample_time = 0.1  # FIXME: this should be a global constant


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
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


def main_callback(nav: Navigator):
    if not nav.cap.isOpened():
        nav.is_running = False
        return

    ret, frame = nav.cap.read()
    if not ret:
        nav.is_running = False
        print('Cannot read frame')
        return

    cv2.undistort(frame, nav.camera_matrix, nav.distortion_coeffs, dst=nav.frame_undistorted,
                  newCameraMatrix=nav.new_camera_matrix)
    nav.img_undistorted[:] = nav.frame_undistorted

    corners, ids, rejected = nav.detector.detectMarkers(nav.frame_undistorted)

    map_found, map_corners = detect_map(corners, ids)
    text_y = 30
    if not map_found:
        cv2.putText(nav.img_undistorted, 'Map not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
        text_y += 30
        nav.img_map[:] = nav.base_img_map
    else:
        cv2.putText(nav.img_undistorted, 'Map detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
        text_y += 30
        for corner in map_corners:
            cv2.drawMarker(nav.img_undistorted, position=corner.astype(np.int32), color=(0, 0, 255),
                           markerType=cv2.MARKER_CROSS, thickness=2)

        matrix = get_perspective_transform(map_corners, nav.map_width_px, nav.map_height_px)
        cv2.warpPerspective(nav.frame_undistorted, matrix, dsize=(nav.map_width_px, nav.map_height_px),
                            dst=nav.frame_map)
        nav.base_img_map[:] = nav.frame_map
        nav.img_map[:] = nav.frame_map

    corners, ids, rejected = nav.detector.detectMarkers(nav.frame_map)

    robot_found, robot_position, robot_direction = detect_robot(corners, ids)
    target_found, target_position = detect_target(corners, ids)

    text_y = 30
    if not robot_found:
        cv2.putText(nav.img_map, 'Robot not detected', org=(10, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
        text_y += 30
    else:
        cv2.putText(nav.img_map, 'Robot detected', org=(10, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
        text_y += 30

        pos = robot_position.astype(np.int32)
        tip = (robot_position + robot_direction).astype(np.int32)
        cv2.arrowedLine(nav.img_map, pos, tip, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA,
                        tipLength=0.5)
        cv2.drawMarker(nav.img_map, position=pos, color=(0, 0, 255), thickness=2, markerSize=10,
                       markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)

        robot_position_world = transform_affine(nav.image_to_world, robot_position)
        robot_x = robot_position_world.item(0)
        robot_y = robot_position_world.item(1)
        robot_theta = np.arctan2(-robot_direction[0], -robot_direction[1])
        outline = get_robot_outline(robot_x, robot_y, robot_theta).astype(np.int32)
        outline = np.array([transform_affine(nav.world_to_image, pt) for pt in outline], dtype=np.int32)
        cv2.polylines(nav.img_map, [outline], isClosed=True, color=(0, 0, 255), thickness=2,
                      lineType=cv2.LINE_AA)

        measurements = np.array([[robot_x], [robot_y], [robot_theta]])
        if nav.loop_index == 0:
            nav.prev_x_est[:] = measurements
            if len(nav.path_world) != 0:
                nav.dist_error = np.sqrt(
                    (nav.path_world[0, 0] - measurements[0, 0]) ** 2 + (nav.path_world[0, 1] - measurements[1, 0]) ** 2)
                nav.angle_error = np.rad2deg(nav.path_world[0, 2] - measurements[2, 0])

        new_x_est, new_P_est = Algorithm_EKF(measurements, nav.prev_x_est, nav.prev_P_est, nav.prev_input)
        nav.prev_x_est = new_x_est
        nav.prev_P_est = new_P_est

        if len(nav.path_world) != 0:
            nav.prev_x_est = nav.prev_x_est.tolist()
            nav.prev_x_est[2] = np.rad2deg(nav.prev_x_est[2])
            goal_state = [nav.path_world[0, 0], nav.path_world[1, 0], np.rad2deg(nav.path_world[2, 0])]
            nav.prev_input, nav.switch, nav.angle_error, nav.dist_error = control(nav.prev_x_est, goal_state,
                                                                                  nav.switch, nav.angle_error,
                                                                                  nav.dist_error, nav.sample_time)
            nav.prev_x_est = np.array(nav.prev_x_est)
            nav.prev_x_est[2] = np.deg2rad(nav.prev_x_est[2])
            nav.node.send_set_variables(move_robot(nav.prev_input[0], nav.prev_input[1]))

        print(f'x={new_x_est}, Sigma={new_P_est}')

    if not target_found:
        cv2.putText(nav.img_map, 'Target not detected', org=(10, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
        text_y += 30
    else:
        cv2.putText(nav.img_map, 'Target detected', org=(10, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
        text_y += 30
        cv2.drawMarker(nav.img_map, position=target_position.astype(np.int32), color=(0, 255, 0),
                       markerSize=10,
                       markerType=cv2.MARKER_CROSS)

    if nav.graph is not None:
        draw_static_graph(nav.img_map, nav.graph, nav.regions)
        if nav.free_source is not None and nav.free_target is not None:
            draw_path(nav.img_map, nav.graph, nav.path, nav.source_position, nav.free_source,
                      nav.stored_target_position, nav.free_target)

    key = cv2.waitKey(1) & 0xff
    if key == 27:
        nav.is_running = False
        return
    elif key == ord('m'):
        nav.regions, nav.graph = build_static_graph(nav.frame_map,
                                                    nav.dilation_radius_px,
                                                    robot_position if robot_found else None,
                                                    nav.robot_radius_px,
                                                    target_position if target_found else None,
                                                    nav.target_radius_px,
                                                    nav.marker_size_px,
                                                    nav.map_width_px,
                                                    nav.map_height_px)
    elif key == ord('u'):
        if robot_found and target_found:
            nav.source_position = robot_position
            nav.stored_target_position = target_position
            nav.free_source, nav.free_target = update_graph(nav.graph, nav.regions, nav.source_position,
                                                            nav.stored_target_position)
            nav.path = dijkstra(nav.graph.adjacency, Graph.SOURCE, Graph.TARGET)
            nav.path_world = np.empty((len(nav.path) - 1, 3))
            nav.path_world[:, :2] = np.array(
                [transform_affine(nav.world_to_image, nav.graph.vertices[nav.path[i]]) for i in
                 range(1, len(nav.path))])
            nav.path_world[:, 2] = np.arctan2(nav.path_world[:, 1], nav.path_world[:, 0])

    cv2.imshow('Undistorted frame', nav.img_undistorted)
    cv2.imshow('Map', nav.img_map)

    nav.loop_index += 1


async def main():
    nav = Navigator()

    client = ClientAsync()
    nav.node = await client.wait_for_node()
    await nav.node.lock()

    nav.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_width = 960
    frame_height = 720
    nav.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    nav.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    map_width_mm = 1000 - 25
    map_height_mm = 696 - 25
    map_width_px = 900
    map_height_px = int(map_width_px * map_height_mm / map_width_mm)
    nav.image_to_world = get_image_to_world_matrix(map_width_px, map_height_px, map_width_mm, map_height_mm)
    nav.world_to_image = get_world_to_image_matrix(map_width_mm, map_height_mm, map_width_px, map_height_px)
    nav.map_width_px = map_width_px
    nav.map_height_px = map_height_px

    # Convention: in main(), all frame_* images are destined to image processing, and have no extra drawings on them.
    # All img_* images are the ones which have extra text, markers, etc. drawn on them.

    # Undistorted
    nav.frame_undistorted = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    nav.img_undistorted = np.zeros_like(nav.frame_undistorted)

    # Undistorted, perspective corrected
    nav.frame_map = np.zeros((map_height_px, map_width_px, 3), dtype=np.uint8)
    nav.img_map = np.zeros_like(nav.frame_map)
    nav.base_img_map = np.zeros_like(nav.frame_map)

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    nav.detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # camera_matrix, distortion_coeffs = calibrate_camera(frame_width, frame_height)
    # store_to_json('camera.json', camera_matrix, distortion_coeffs)
    # return
    nav.camera_matrix, nav.distortion_coeffs = load_from_json('camera.json')
    nav.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(nav.camera_matrix, nav.distortion_coeffs,
                                                               (map_width_px, map_height_px), 0,
                                                               (map_width_px, map_height_px))

    nav.dilation_radius_px = int((ROBOT_RADIUS + 10) / map_width_mm * map_width_px)
    nav.robot_radius_px = int((ROBOT_RADIUS + 20) / map_width_mm * map_width_px)
    nav.target_radius_px = int((TARGET_RADIUS + 20) / map_width_mm * map_width_px)
    nav.marker_size_px = int((MARKER_SIZE + 5) / map_width_mm * map_width_px)

    timer = RepeatTimer(0.1, main_callback, args=[nav])
    timer.start()

    while nav.is_running:
        time.sleep(1.0 / 30.0)

    nav.cap.release()
    cv2.destroyAllWindows()

    await nav.node.unlock()


if __name__ == '__main__':
    asyncio.run(main())
