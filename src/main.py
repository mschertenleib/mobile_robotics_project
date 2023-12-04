import asyncio
from threading import Timer

from tdmclient.atranspiler import ATranspiler

from camera_calibration import *
from controller import *
from global_map import *
from image_processing import *
from kalman_filter import *
from local_navigation import *
from parameters import *
from threaded_capture import *


class Navigator:
    """
    Holds the persistent state needed in the navigation timer callback
    """

    def __init__(self):
        self.node = None
        self.first_estimate = True
        self.last_sample_time = time.time()
        self.frame_map = None
        self.img_map = None
        self.detector = None
        self.image_to_world = None
        self.world_to_image = None
        self.regions = None
        self.graph = None
        self.path_image: list[int] = []
        self.path_world = None
        self.path_index = 0
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
        # P_est at convergence during straight line test
        self.prev_P_est = np.array([[4.12148917e-02, -1.07933653e-04, 4.21480900e-04],
                                    [-1.07933653e-04, 4.04040766e-02, -5.94141187e-05],
                                    [4.21480900e-04, -5.94141187e-05, 3.06223793e-02]])
        self.prev_input = np.zeros(2)
        self.angle_error = 0.0
        self.dist_error = 0.0
        self.switch = 0


class LocalNavigator:
    def __init__(self, client, node):
        self.client = client
        self.node = node
        self.spped_gain = 2
        self.motor_speed = 100  # speed of the robot
        self.LTobst = 5  # low obstacle threshold to switch state 1->0
        self.HTobst = 13  # high obstacle threshold to switch state 0->1
        self.obst_gain = 15  # /100 (actual gain: 15/100=0.15)
        self.state = 0  # Actual state of the robot: 0->global navigation, 1->obstacle avoidance
        self.case = 0  # Actual case of obstacle avoidance: 0-> left obstacle, 1-> right obstacle, 2-> obstacle in front
        # we randomly generate the first bypass choice of the robot when he encounters an obstacle in front of him
        self.side = bool(np.random.randint(2))
        self.rotation_time = 1  # 1ms time before rotation
        self.step_back_time = 1  # 1ms time during which i step back
        self.prox_horizontal = [0] * 5


class RepeatTimer(Timer):
    def run(self):
        wait_time = self.interval
        while not self.finished.wait(wait_time):
            time_start_function = time.time()
            self.function(*self.args, **self.kwargs)
            time_end_function = time.time()
            wait_time = self.interval - (time_end_function - time_start_function)


async def compile_run_python_for_thymio(node):
    await node.register_events([("requestspeed", 2)])

    with open('thymio_program.py', 'r') as file:
        thymio_program_python = file.read()
        thymio_program_aseba = ATranspiler.simple_transpile(thymio_program_python)
        compilation_result = await node.compile(thymio_program_aseba)
        if compilation_result is None:
            print('Compilation success')
        else:
            print('Compilation error :', compilation_result)
            return False
        await node.run()

    return True


def build_static_graph(img: np.ndarray, robot_position: typing.Optional[np.ndarray],
                       target_position: typing.Optional[np.ndarray]) -> tuple[list[list[np.ndarray]], Graph]:
    # Note: the minimum distance to any obstacle is 'dilation_size_px - approx_poly_epsilon'
    approx_poly_epsilon = 2
    obstacle_mask = get_obstacle_mask(img, DILATION_RADIUS_PX, robot_position, ROBOT_RADIUS_PX, target_position,
                                      TARGET_RADIUS_PX, MARKER_SIZE_PX, MAP_WIDTH_PX, MAP_HEIGHT_PX)
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
    nav.img_map[:] = nav.frame_map

    corners, ids, rejected = nav.detector.detectMarkers(nav.frame_map)

    nav.robot_found, nav.robot_position, nav.robot_direction = detect_robot(corners, ids)
    nav.target_found, nav.target_position = detect_target(corners, ids)

    if nav.robot_found:
        pos = nav.robot_position.astype(np.int32)
        tip = (nav.robot_position + nav.robot_direction).astype(np.int32)
        cv2.arrowedLine(nav.img_map, pos, tip, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA,
                        tipLength=0.5)
        cv2.drawMarker(nav.img_map, position=pos, color=(0, 0, 255), thickness=2, markerSize=10,
                       markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)

        robot_position_world = transform_affine(nav.image_to_world, nav.robot_position)
        robot_x = robot_position_world.item(0)
        robot_y = robot_position_world.item(1)
        robot_theta = np.arctan2(-nav.robot_direction[0], -nav.robot_direction[1])

        outline = get_robot_outline(robot_x, robot_y, robot_theta).astype(np.int32)
        outline = np.array([transform_affine(nav.world_to_image, pt) for pt in outline], dtype=np.int32)
        cv2.polylines(nav.img_map, [outline], isClosed=True, color=(0, 0, 255), thickness=2,
                      lineType=cv2.LINE_AA)

        measurements = np.array([[robot_x], [robot_y], [robot_theta]])
        if nav.first_estimate:
            nav.prev_x_est[:] = measurements
            if nav.path_world is not None and len(nav.path_world) != 0:
                nav.dist_error = np.sqrt(
                    (nav.path_world[0, 0] - measurements[0, 0]) ** 2 + (nav.path_world[0, 1] - measurements[1, 0]) ** 2)
                nav.angle_error = np.rad2deg(nav.path_world[0, 2] - measurements[2, 0])
            nav.first_estimate = False

        speed_left = int(nav.node["motor.left.speed"])
        speed_right = int(nav.node["motor.right.speed"])
        prev_input = np.array([speed_left * MMS_PER_MOTOR_SPEED, speed_right * MMS_PER_MOTOR_SPEED])
        new_x_est, new_P_est = kalman_filter(measurements, nav.prev_x_est, nav.prev_P_est, prev_input)
        nav.prev_x_est = new_x_est
        nav.prev_P_est = new_P_est

        estimated_state = new_x_est.flatten()
        outline = get_robot_outline(estimated_state[0], estimated_state[1], estimated_state[2]).astype(np.int32)
        outline = np.array([transform_affine(nav.world_to_image, pt) for pt in outline], dtype=np.int32)
        cv2.polylines(nav.img_map, [outline], isClosed=True, color=(192, 64, 64), thickness=2,
                      lineType=cv2.LINE_AA)

        if nav.path_world is not None and len(nav.path_world) != 0:
            nav.prev_x_est = nav.prev_x_est.tolist()
            goal_state = [nav.path_world[nav.path_index, 0], nav.path_world[nav.path_index, 1]]

            input_left, input_right, goal_reached = astolfi_control(np.array(nav.prev_x_est).flatten(), goal_state)
            if goal_reached and nav.path_index < len(nav.path_world) - 1:
                nav.path_index += 1

            nav.prev_input[:] = input_left, input_right
            nav.prev_x_est = np.array(nav.prev_x_est)
            u_l = np.clip(int(nav.prev_input[0] / MMS_PER_MOTOR_SPEED), -500, 500)
            u_r = np.clip(int(nav.prev_input[1] / MMS_PER_MOTOR_SPEED), -500, 500)
            nav.node.send_send_events({"requestspeed": [u_l, u_r]})

        print(f'X estimate = {new_x_est.flatten()}')

    if nav.target_found:
        cv2.drawMarker(nav.img_map, position=nav.target_position.astype(np.int32), color=(0, 255, 0),
                       markerSize=10, markerType=cv2.MARKER_CROSS, thickness=2, line_type=cv2.LINE_AA)
        cv2.circle(nav.img_map, center=nav.target_position.astype(np.int32), radius=50, color=(0, 255, 0), thickness=2,
                   lineType=cv2.LINE_AA)

    if nav.graph is not None:
        draw_static_graph(nav.img_map, nav.graph, nav.regions)
        if len(nav.path_image) >= 2 and nav.free_robot_position is not None and nav.free_target_position is not None:
            draw_path(nav.img_map, nav.graph, nav.path_image, nav.stored_robot_position, nav.free_robot_position,
                      nav.stored_target_position, nav.free_target_position)

    cv2.imshow('Map', nav.img_map)
    cv2.waitKey(1)


def run_local_navigation(loc_nav: LocalNavigator):
    prox_horizontal = [0] * 5
    for i in range(5):
        prox_horizontal[i] = list(loc_nav.node.v.prox.horizontal)[i]
    obst = [prox_horizontal[0], prox_horizontal[1], prox_horizontal[2], prox_horizontal[3], prox_horizontal[4]]

    asyncio.run(avoid_obstacles(loc_nav.node, loc_nav.client, loc_nav.state, loc_nav.side, obst, loc_nav.HTobst,
                                loc_nav.LTobst, loc_nav.motor_speed, loc_nav.step_back_time, loc_nav.spped_gain,
                                loc_nav.rotation_time))


async def main():
    nav = Navigator()

    video_thread = VideoThread(FRAME_WIDTH, FRAME_HEIGHT)

    nav.image_to_world = get_image_to_world_matrix(MAP_WIDTH_PX, MAP_HEIGHT_PX, MAP_WIDTH_MM, MAP_HEIGHT_MM)
    nav.world_to_image = get_world_to_image_matrix(MAP_WIDTH_MM, MAP_HEIGHT_MM, MAP_WIDTH_PX, MAP_HEIGHT_PX)

    # Convention: in main(), all frame_* images are destined to image processing, and have no extra drawings on them.
    # All img_* images are the ones which have extra text, markers, etc. drawn on them.

    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame_undistorted = np.zeros_like(frame)
    img_undistorted = np.zeros_like(frame)

    nav.frame_map = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), dtype=np.uint8)
    nav.img_map = np.zeros_like(nav.frame_map)

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    nav.detector = detector

    # camera_matrix, distortion_coeffs = calibrate_camera(frame_width, frame_height)
    # store_to_json('camera.json', camera_matrix, distortion_coeffs)
    # return
    camera_matrix, distortion_coeffs = load_from_json('camera.json')
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs,
                                                           (MAP_WIDTH_PX, MAP_HEIGHT_PX), 0,
                                                           (MAP_WIDTH_PX, MAP_HEIGHT_PX))

    client = ClientAsync()
    nav.node = await client.wait_for_node()
    await nav.node.lock()

    compile_ok = await compile_run_python_for_thymio(nav.node)
    if not compile_ok:
        await nav.node.unlock()

    await nav.node.wait_for_variables()

    timer = RepeatTimer(SAMPLING_TIME, run_navigation, args=[nav])
    timer.start()

    loc_nav = LocalNavigator(client, nav.node)
    local_nav_timer = RepeatTimer(0.1, run_local_navigation, args=[loc_nav])
    # local_nav_timer.start()

    while True:
        # Let the client handle its work
        await client.sleep(0.005)

        video_thread.get_frame(frame)

        cv2.undistort(frame, camera_matrix, distortion_coeffs, dst=frame_undistorted,
                      newCameraMatrix=new_camera_matrix)
        img_undistorted[:] = frame_undistorted

        corners, ids, rejected = detector.detectMarkers(frame_undistorted)

        map_found, map_corners = detect_map(corners, ids)
        if not map_found:
            cv2.putText(img_undistorted, 'Map not detected', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 64, 192), lineType=cv2.LINE_AA)
        else:
            cv2.putText(img_undistorted, 'Map detected', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(64, 192, 64), lineType=cv2.LINE_AA)
            for corner in map_corners:
                cv2.drawMarker(img_undistorted, position=corner.astype(np.int32), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, thickness=2)

            matrix = get_perspective_transform(map_corners, MAP_WIDTH_PX, MAP_HEIGHT_PX)
            cv2.warpPerspective(frame_undistorted, matrix, dsize=(MAP_WIDTH_PX, MAP_HEIGHT_PX),
                                dst=nav.frame_map)

        cv2.imshow('Undistorted frame', img_undistorted)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord('m'):
            nav.regions, nav.graph = build_static_graph(nav.frame_map,
                                                        nav.robot_position if nav.robot_found else None,
                                                        nav.target_position if nav.target_found else None)
        elif key == ord('u'):
            if nav.robot_found and nav.target_found:
                nav.stored_robot_position = nav.robot_position
                nav.stored_target_position = nav.target_position
                nav.free_robot_position, nav.free_target_position = update_graph(nav.graph, nav.regions,
                                                                                 nav.stored_robot_position,
                                                                                 nav.stored_target_position)
                nav.path_image = dijkstra(nav.graph.adjacency, Graph.SOURCE, Graph.TARGET)
                if len(nav.path_image) >= 2:
                    nav.path_world = np.empty((len(nav.path_image) - 1, 2))
                    nav.path_world[:, :2] = np.array(
                        [transform_affine(nav.image_to_world, nav.graph.vertices[nav.path_image[i]]) for i in
                         range(1, len(nav.path_image))])
                    nav.path_index = 0

    timer.cancel()
    local_nav_timer.cancel()

    await nav.node.stop()
    await nav.node.unlock()

    video_thread.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main())
