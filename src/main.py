import asyncio
from threading import Timer

from tdmclient import ClientAsync

from camera_calibration import *
from controller import *
from global_map import *
from image_processing import *
from kalman_filter import *
from parameters import *


class Navigator:
    """
    Holds the persistent state needed in the navigation timer callback
    """

    def __init__(self):
        self.node = None
        self.is_running = True
        self.loop_index = 0
        self.frame_map = None
        self.img_map = None
        self.detector = None
        self.image_to_world = None
        self.world_to_image = None
        self.map_width_px = 0
        self.map_height_px = 0
        self.dilation_radius_px = 0
        self.robot_radius_px = 0
        self.target_radius_px = 0
        self.marker_size_px = 0
        self.regions = None
        self.graph = None
        self.path_image: list[int] = []
        self.path_world = None
        self.robot_found = False
        self.robot_position = None
        self.robot_direction = None
        self.target_found = False
        self.target_position = None
        self.stored_robot_position = np.zeros(2)
        self.free_robot_position = np.zeros(2)
        self.stored_target_position = np.zeros(2)
        self.free_target_position = np.zeros(2)
        self.prev_x_est = np.zeros((3, 1))
        self.prev_P_est = 1000 * np.ones(3)
        self.prev_input = np.zeros(2)
        self.angle_error = 0.0
        self.dist_error = 0.0
        self.switch = 0


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


def run_navigation(nav: Navigator):
    corners, ids, rejected = nav.detector.detectMarkers(nav.frame_map)

    nav.robot_found, nav.robot_position, nav.robot_direction = detect_robot(corners, ids)
    nav.target_found, nav.target_position = detect_target(corners, ids)

    if nav.robot_found:
        robot_position_world = transform_affine(nav.image_to_world, nav.robot_position)
        robot_x = robot_position_world.item(0)
        robot_y = robot_position_world.item(1)
        robot_theta = np.arctan2(-nav.robot_direction[0], -nav.robot_direction[1])
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

        if nav.path_world is not None and len(nav.path_world) != 0:
            nav.prev_x_est = nav.prev_x_est.tolist()
            nav.prev_x_est[2] = np.rad2deg(nav.prev_x_est[2])
            goal_state = [nav.path_world[0, 0], nav.path_world[1, 0], np.rad2deg(nav.path_world[2, 0])]
            nav.prev_input, nav.switch, nav.angle_error, nav.dist_error = control(nav.prev_x_est, goal_state,
                                                                                  nav.switch, nav.angle_error,
                                                                                  nav.dist_error, SAMPLING_TIME)
            nav.prev_x_est = np.array(nav.prev_x_est)
            nav.prev_x_est[2] = np.deg2rad(nav.prev_x_est[2])
            nav.node.send_set_variables(move_robot(nav.prev_input[0], nav.prev_input[1]))

        print(f'x={new_x_est}, Sigma={new_P_est}')

    if nav.robot_found:
        pos = nav.robot_position.astype(np.int32)
        tip = (nav.robot_position + nav.robot_direction).astype(np.int32)
        cv2.arrowedLine(nav.img_map, pos, tip, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA,
                        tipLength=0.5)
        cv2.drawMarker(nav.img_map, position=pos, color=(0, 0, 255), thickness=2, markerSize=10,
                       markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)

    if nav.target_found:
        cv2.drawMarker(nav.img_map, position=nav.target_position.astype(np.int32), color=(0, 255, 0),
                       markerSize=10,
                       markerType=cv2.MARKER_CROSS)

    if nav.graph is not None:
        draw_static_graph(nav.img_map, nav.graph, nav.regions)
        if nav.free_robot_position is not None and nav.free_target_position is not None:
            draw_path(nav.img_map, nav.graph, nav.path_image, nav.stored_robot_position, nav.free_robot_position,
                      nav.stored_target_position, nav.free_target_position)

    cv2.imshow('Map', nav.img_map)
    cv2.waitKey(1)

    nav.loop_index += 1


async def main():
    nav = Navigator()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_width = 960
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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
    frame_undistorted = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    img_undistorted = np.zeros_like(frame_undistorted)

    # Undistorted, perspective corrected
    nav.frame_map = np.zeros((map_height_px, map_width_px, 3), dtype=np.uint8)
    nav.img_map = np.zeros_like(nav.frame_map)

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    nav.detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # camera_matrix, distortion_coeffs = calibrate_camera(frame_width, frame_height)
    # store_to_json('camera.json', camera_matrix, distortion_coeffs)
    # return
    camera_matrix, distortion_coeffs = load_from_json('camera.json')
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs,
                                                           (map_width_px, map_height_px), 0,
                                                           (map_width_px, map_height_px))

    nav.dilation_radius_px = int((ROBOT_RADIUS + 10) / map_width_mm * map_width_px)
    nav.robot_radius_px = int((ROBOT_RADIUS + 20) / map_width_mm * map_width_px)
    nav.target_radius_px = int((TARGET_RADIUS + 20) / map_width_mm * map_width_px)
    nav.marker_size_px = int((MARKER_SIZE + 5) / map_width_mm * map_width_px)

    client = ClientAsync()
    nav.node = await client.wait_for_node()
    await nav.node.lock()

    timer = RepeatTimer(SAMPLING_TIME, run_navigation, args=[nav])
    timer.start()

    while cap.isOpened() and nav.is_running:
        ret, frame = cap.read()
        if not ret:
            nav.is_running = False
            print('Cannot read frame')
            break

        cv2.undistort(frame, camera_matrix, distortion_coeffs, dst=frame_undistorted,
                      newCameraMatrix=new_camera_matrix)
        img_undistorted[:] = frame_undistorted

        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        map_found, map_corners = detect_map(corners, ids)
        text_y = 30
        if not map_found:
            cv2.putText(img_undistorted, 'Map not detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
            text_y += 30
        else:
            cv2.putText(img_undistorted, 'Map detected', org=(10, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
            text_y += 30
            for corner in map_corners:
                cv2.drawMarker(img_undistorted, position=corner.astype(np.int32), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, thickness=2)

            matrix = get_perspective_transform(map_corners, map_width_px, map_height_px)
            cv2.warpPerspective(frame_undistorted, matrix, dsize=(map_width_px, map_height_px),
                                dst=nav.frame_map)
            nav.img_map[:] = nav.frame_map

        cv2.imshow('Undistorted frame', img_undistorted)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            nav.is_running = False
            break
        elif key == ord('m'):
            nav.regions, nav.graph = build_static_graph(nav.frame_map,
                                                        nav.dilation_radius_px,
                                                        nav.robot_position if nav.robot_found else None,
                                                        nav.robot_radius_px,
                                                        nav.target_position if nav.target_found else None,
                                                        nav.target_radius_px,
                                                        nav.marker_size_px,
                                                        nav.map_width_px,
                                                        nav.map_height_px)
        elif key == ord('u'):
            if nav.robot_found and nav.target_found:
                nav.stored_robot_position = nav.robot_position
                nav.stored_target_position = nav.target_position
                nav.free_robot_position, nav.free_target_position = update_graph(nav.graph, nav.regions,
                                                                                 nav.stored_robot_position,
                                                                                 nav.stored_target_position)
                nav.path_image = dijkstra(nav.graph.adjacency, Graph.SOURCE, Graph.TARGET)
                nav.path_world = np.empty((len(nav.path_image) - 1, 3))
                nav.path_world[:, :2] = np.array(
                    [transform_affine(nav.world_to_image, nav.graph.vertices[nav.path_image[i]]) for i in
                     range(1, len(nav.path_image))])
                nav.path_world[:, 2] = np.arctan2(nav.path_world[:, 1], nav.path_world[:, 0])

    timer.cancel()

    await nav.node.unlock()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main())
