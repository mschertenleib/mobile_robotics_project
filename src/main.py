from threading import Timer

import dearpygui.dearpygui as dpg
from tdmclient import aw
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

    def __init__(self, client, node):
        self.start_time = time.time()
        self.node = None
        self.requires_first_measurement = True
        self.map_found = False
        self.should_follow_path = False
        # BGR u8 undistorted and perspective corrected image destined to image processing
        self.frame_map = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), dtype=np.uint8)
        # BGR u8 image destined to be displayed
        self.img_map = np.zeros_like(self.frame_map)
        # RGBA f32 image for displaying with dearpygui
        self.map_rgba_f32 = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 4), dtype=np.float32)
        self.map_rgba_f32[:, :, 3] = 1.0
        self.detector = None
        self.image_to_world = get_image_to_world_matrix(MAP_WIDTH_PX, MAP_HEIGHT_PX, MAP_WIDTH_MM, MAP_HEIGHT_MM)
        self.world_to_image = get_world_to_image_matrix(MAP_WIDTH_MM, MAP_HEIGHT_MM, MAP_WIDTH_PX, MAP_HEIGHT_PX)
        self.regions = None
        self.graph = None
        self.path_image_space: list[int] = []
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
        self.prev_P_est = KALMAN_Q
        # Data for the plots
        self.sample_time_history = []
        self.estimated_pose_history = []
        self.measured_pose_history = []
        # Local navigation
        self.client = client
        self.node = node
        self.motor_speed = 200  # speed of the robot
        self.LTobst = 5  # low obstacle threshold to switch state 1->0
        self.HTobst = 17  # high obstacle threshold to switch state 0->1
        self.obst_gain = 7  # /100 (actual gain: 15/100=0.15)
        self.num_samples_since_last_obstacle = -1
        self.was_avoiding = False  # Actual state of the robot: 0->global navigation, 1->obstacle avoidance
        self.case = 0  # Actual case of obstacle avoidance: 0-> left obstacle, 1-> right obstacle, 2-> obstacle in front
        # we randomly generate the first bypass choice of the robot when he encounters an obstacle in front of him
        self.side = bool(np.random.randint(2))
        self.rotation_time = 1  # 1ms time before rotation
        self.step_back_time = 1  # 1ms time during which i step back


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
            print('Aseba compilation success')
        else:
            print('Aseba compilation error :', compilation_result)
            return False
        await node.run()

    return True


def build_static_graph(nav: Navigator):
    obstacle_mask = get_obstacle_mask(nav.frame_map, nav.robot_position if nav.robot_found else None,
                                      nav.target_position if nav.target_found else None)
    # Note: the minimum distance to any obstacle is 'DILATION_SIZE_PX - approx_poly_epsilon'
    approx_poly_epsilon = 2
    nav.regions = extract_contours(obstacle_mask, approx_poly_epsilon)
    if len(nav.regions) > 0:
        nav.graph = build_graph(nav.regions)


def build_dynamic_graph_and_plan_path(nav: Navigator):
    if nav.graph is not None and nav.robot_found and nav.target_found:
        nav.stored_robot_position = nav.robot_position
        nav.stored_target_position = nav.target_position
        nav.free_robot_position, nav.free_target_position = update_graph(nav.graph, nav.regions,
                                                                         nav.stored_robot_position,
                                                                         nav.stored_target_position)
        nav.path_image_space = dijkstra(nav.graph.adjacency, Graph.SOURCE, Graph.TARGET)

        if len(nav.path_image_space) >= 2:
            nav.path_world = np.array(
                [transform_affine(nav.image_to_world, nav.graph.vertices[nav.path_image_space[i]]) for i in
                 range(1, len(nav.path_image_space))])
        else:
            nav.path_world = None
        nav.path_index = 0


def callback_button_build_map(sender, app_data, user_data):
    assert isinstance(user_data, Navigator)
    build_static_graph(user_data)
    build_dynamic_graph_and_plan_path(user_data)


def callback_button_go(sender, app_data, user_data):
    assert isinstance(user_data, Navigator)
    user_data.should_follow_path = True


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


def resize_image_to_fit_window(window: Union[int, str], image: Union[int, str], image_aspect_ratio: float):
    window_width, window_height = dpg.get_item_rect_size(window)
    if window_width <= 0 or window_height <= 0:
        return

    # Use the item position within the window to deduce the available size in the window
    image_pos_x, image_pos_y = dpg.get_item_pos(image)
    window_content_width = window_width - 2 * image_pos_x
    window_content_height = window_height - image_pos_y - image_pos_x
    if window_content_width <= 0 or window_content_height <= 0:
        return

    # Make the image as big as possible while keeping its aspect ratio
    image_width, image_height = window_content_width, window_content_height
    if window_content_width / window_content_height >= image_aspect_ratio:
        image_width = int(image_height * image_aspect_ratio)
    else:
        image_height = int(image_width / image_aspect_ratio)

    dpg.set_item_width(image, image_width)
    dpg.set_item_height(image, image_height)


def update_plots(nav: Navigator):
    if len(nav.sample_time_history) < 2:
        return

    num_samples_to_plot = dpg.get_value('tag_samples_slider')
    num_samples_to_plot = min(num_samples_to_plot, len(nav.sample_time_history))
    time_data = nav.sample_time_history
    estimated_pose_arr = np.array(nav.estimated_pose_history)
    measured_pose_arr = np.array(nav.measured_pose_history)

    dpg.set_value('tag_series_x_est', [time_data, estimated_pose_arr[:, 0].tolist()])
    dpg.set_value('tag_series_y_est', [time_data, estimated_pose_arr[:, 1].tolist()])
    dpg.set_value('tag_series_theta_est', [time_data, estimated_pose_arr[:, 2].tolist()])
    dpg.set_value('tag_series_x_meas', [time_data, measured_pose_arr[:, 0].tolist()])
    dpg.set_value('tag_series_y_meas', [time_data, measured_pose_arr[:, 1].tolist()])
    dpg.set_value('tag_series_theta_meas', [time_data, measured_pose_arr[:, 2].tolist()])

    if dpg.get_value('tag_checkbox_autofit'):
        # Set limits on the time axis (will automatically change the time axis of the other plot, since they are linked)
        dpg.set_axis_limits('tag_plot_xy_axis_x', time_data[-num_samples_to_plot], time_data[-1])

        # Set limits on the y axes
        min_pos = min(np.min(estimated_pose_arr[-num_samples_to_plot:, :2]),
                      np.min(measured_pose_arr[-num_samples_to_plot:, :2]))
        max_pos = max(np.max(estimated_pose_arr[-num_samples_to_plot:, :2]),
                      np.max(measured_pose_arr[-num_samples_to_plot:, :2]))
        min_theta = min(np.min(estimated_pose_arr[-num_samples_to_plot:, 2]),
                        np.min(measured_pose_arr[-num_samples_to_plot:, 2]))
        max_theta = max(np.max(estimated_pose_arr[-num_samples_to_plot:, 2]),
                        np.max(measured_pose_arr[-num_samples_to_plot:, 2]))
        pos_range = max_pos - min_pos
        theta_range = max_theta - min_theta
        ymin_xy = min_pos - pos_range * 0.1 if pos_range > 0 else min_pos - 1
        ymax_xy = max_pos + pos_range * 0.1 if pos_range > 0 else max_pos + 1
        ymin_theta = min_theta - theta_range * 0.1 if theta_range > 0 else min_theta - 1
        ymax_theta = max_theta + theta_range * 0.1 if theta_range > 0 else max_theta + 1
        dpg.set_axis_limits('tag_plot_xy_axis_y', ymin_xy, ymax_xy)
        dpg.set_axis_limits('tag_plot_theta_axis_y', ymin_theta, ymax_theta)
    else:
        dpg.set_axis_limits_auto('tag_plot_xy_axis_x')
        dpg.set_axis_limits_auto('tag_plot_xy_axis_y')
        dpg.set_axis_limits_auto('tag_plot_theta_axis_y')


def build_interface(nav: Navigator):
    dpg.configure_app(docking=True, docking_space=True, init_file='imgui.ini', auto_save_init_file=True)
    dpg.create_viewport(title='Robot interface', width=1280, height=720)

    with dpg.texture_registry():
        default_frame_data = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.float32)
        default_frame_data[:, :, 3] = 1.0
        default_map_data = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 4), dtype=np.float32)
        default_map_data[:, :, 3] = 1.0
        dpg.add_raw_texture(width=FRAME_WIDTH, height=FRAME_HEIGHT, default_value=default_frame_data,
                            format=dpg.mvFormat_Float_rgba,
                            tag='tag_frame_texture')
        dpg.add_raw_texture(width=MAP_WIDTH_PX, height=MAP_HEIGHT_PX, default_value=default_map_data,
                            format=dpg.mvFormat_Float_rgba,
                            tag='tag_map_texture')

    with dpg.window(label='Camera', tag='tag_camera_window', no_close=True):
        dpg.add_image(texture_tag='tag_frame_texture', tag='tag_frame_image')

    with dpg.window(label='Map', tag='tag_map_window', no_close=True):
        dpg.add_button(label='Build map', callback=callback_button_build_map, user_data=nav)
        dpg.add_button(label='GO!', callback=callback_button_go, user_data=nav)
        dpg.add_image(texture_tag='tag_map_texture', tag='tag_map_image')

    with dpg.window(label='Plots', tag='tag_plot_window', no_close=True):
        dpg.add_slider_int(label='Samples to plot', tag='tag_samples_slider', default_value=300, min_value=10,
                           max_value=1200)
        dpg.add_checkbox(label='Auto-fit axes', tag='tag_checkbox_autofit', default_value=True)
        with dpg.subplots(2, 1, label='', width=-1, height=-1, link_all_x=True):
            with dpg.plot(label='XY plot', tag='tag_plot_xy', width=-1, height=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label='Time [s]', tag='tag_plot_xy_axis_x')
                with dpg.plot_axis(dpg.mvYAxis, label='Position [mm]', tag='tag_plot_xy_axis_y'):
                    dpg.add_line_series([], [], label='X (estimated)', tag='tag_series_x_est')
                    dpg.add_line_series([], [], label='Y (estimated)', tag='tag_series_y_est')
                    dpg.add_line_series([], [], label='X (measured)', tag='tag_series_x_meas')
                    dpg.add_line_series([], [], label='Y (measured)', tag='tag_series_y_meas')

            with dpg.plot(label='Theta plot', tag='tag_plot_theta', width=-1, height=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label='Time [s]', tag='tag_plot_theta_axis_x')
                with dpg.plot_axis(dpg.mvYAxis, label='Orientation [°]', tag='tag_plot_theta_axis_y'):
                    dpg.add_line_series([], [], label='Theta (estimated)', tag='tag_series_theta_est')
                    dpg.add_line_series([], [], label='Theta (measured)', tag='tag_series_theta_meas')

    dpg.setup_dearpygui()
    dpg.show_viewport()


def run_navigation(nav: Navigator):
    # Reset the render target with the frame value (the undistorted and perspective corrected one). Note that the frame
    # might not have been updated since the last time, if the map was not detected or if the camera frame rate is slower
    # than the frequency of the calls to run_navigation().
    nav.img_map[:] = nav.frame_map

    measurements = None

    should_replan_path = False

    # There is no point in detecting the robot position if the map was not found, because we would just get its position
    # as it was when the map was detected for the last time, and throw off the state estimator because it is not up to
    # date.
    if nav.map_found:
        corners, ids, rejected = nav.detector.detectMarkers(nav.frame_map)

        nav.robot_found, nav.robot_position, nav.robot_direction = detect_robot(corners, ids)

        nav.target_found, target_position = detect_target(corners, ids)
        if nav.target_found and nav.target_position is not None and np.linalg.norm(
                target_position - nav.target_position) >= TARGET_DELTA_TO_PLAN_PATH_AGAIN_MM:
            should_replan_path = True
        nav.target_position = target_position

        if nav.robot_found:
            robot_position_world = transform_affine(nav.image_to_world, nav.robot_position)
            robot_x = robot_position_world.item(0)
            robot_y = robot_position_world.item(1)
            robot_theta = np.arctan2(-nav.robot_direction[0], -nav.robot_direction[1])

            cv2.drawMarker(nav.img_map, position=nav.robot_position.astype(np.int32), color=(0, 0, 255), thickness=2,
                           markerSize=10,
                           markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)
            outline = get_robot_outline(robot_x, robot_y, robot_theta)
            outline = np.array([transform_affine(nav.world_to_image, pt) for pt in outline], dtype=np.int32)
            cv2.polylines(nav.img_map, [outline], isClosed=True, color=(0, 0, 255), thickness=2,
                          lineType=cv2.LINE_AA)

            measurements = np.array([[robot_x], [robot_y], [robot_theta]])
            if nav.requires_first_measurement:
                nav.prev_x_est[:] = measurements
                nav.requires_first_measurement = False

    # We need at least one measurement from the camera to start estimating our state, else we would not know where to
    # start (the initial position of the robot can be completely random).
    if not nav.requires_first_measurement:
        speed_left = int(nav.node.v.motor.left.speed) * MMS_PER_MOTOR_SPEED
        speed_right = int(nav.node.v.motor.right.speed) * MMS_PER_MOTOR_SPEED
        new_x_est, new_P_est = kalman_filter(measurements, nav.prev_x_est, nav.prev_P_est, speed_left, speed_right)
        if np.linalg.norm(new_x_est.flatten()[:2] - nav.prev_x_est.flatten()[:2]) >= ROBOT_DELTA_TO_PLAN_PATH_AGAIN_MM:
            should_replan_path = True

        nav.prev_x_est = new_x_est
        nav.prev_P_est = new_P_est
        estimated_state = new_x_est.flatten()

        nav.sample_time_history.append(time.time() - nav.start_time)
        if measurements is not None:
            nav.measured_pose_history.append(measurements.flatten())
            nav.measured_pose_history[-1][2] = np.rad2deg(nav.measured_pose_history[-1][2])
        else:
            nav.measured_pose_history.append(np.zeros(3))
        nav.estimated_pose_history.append(estimated_state.copy())
        nav.estimated_pose_history[-1][2] = np.rad2deg(nav.estimated_pose_history[-1][2])
        update_plots(nav)

        cv2.drawMarker(nav.img_map, position=transform_affine(nav.world_to_image, np.array(
            [estimated_state[0], estimated_state[1]])).astype(np.int32),
                       color=(64, 192, 64), thickness=2, markerSize=10,
                       markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA)
        outline = get_robot_outline(estimated_state.item(0), estimated_state.item(1), estimated_state.item(2))
        outline = np.array([transform_affine(nav.world_to_image, pt) for pt in outline], dtype=np.int32)
        cv2.polylines(nav.img_map, [outline], isClosed=True, color=(64, 192, 64), thickness=2,
                      lineType=cv2.LINE_AA)

        if should_replan_path:
            build_dynamic_graph_and_plan_path(nav)

        speed_left_target, speed_right_target = 0, 0

        if nav.should_follow_path and nav.path_world is not None and len(nav.path_world) > 0:
            nav.prev_x_est = nav.prev_x_est.tolist()
            goal_state = [nav.path_world[nav.path_index, 0], nav.path_world[nav.path_index, 1]]

            input_left, input_right, goal_reached = astolfi_control(np.array(nav.prev_x_est).flatten(), goal_state)
            if goal_reached:
                if nav.path_index >= len(nav.path_world) - 1:
                    # Final target reached
                    nav.should_follow_path = False
                else:
                    nav.path_index += 1

            nav.prev_x_est = np.array(nav.prev_x_est)

            speed_left_target = np.clip(int(input_left / MMS_PER_MOTOR_SPEED), -500, 500)
            speed_right_target = np.clip(int(input_right / MMS_PER_MOTOR_SPEED), -500, 500)

        prox_horizontal = [0] * 5
        for i in range(5):
            prox_horizontal[i] = list(nav.node.v.prox.horizontal)[i]

        obst = [prox_horizontal[0], prox_horizontal[1], prox_horizontal[2], prox_horizontal[3], prox_horizontal[4]]

        nav.num_samples_since_last_obstacle = aw(
            avoid_obstacles(nav.node, nav.client, nav.num_samples_since_last_obstacle, nav.side, obst, nav.HTobst,
                            nav.LTobst, nav.motor_speed, nav.obst_gain))
        if 0 <= nav.num_samples_since_last_obstacle <= 3:
            nav.num_samples_since_last_obstacle += 1
        elif nav.num_samples_since_last_obstacle > 3:
            nav.num_samples_since_last_obstacle = -1

        elif nav.num_samples_since_last_obstacle == -1:
            nav.node.send_set_variables(set_motor_speed(speed_left_target, speed_right_target))

        if nav.target_found:
            cv2.drawMarker(nav.img_map, position=nav.target_position.astype(np.int32), color=(0, 255, 0),
                           markerSize=10, markerType=cv2.MARKER_CROSS, thickness=2, line_type=cv2.LINE_AA)
            cv2.circle(nav.img_map, center=nav.target_position.astype(np.int32), radius=TARGET_RADIUS_PX,
                       color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if nav.graph is not None and nav.graph.vertices[Graph.SOURCE].shape[0] == 2:
            draw_static_graph(nav.img_map, nav.graph, nav.regions)
            if nav.path_world is not None and len(
                    nav.path_image_space) >= 2 and nav.free_robot_position is not None and nav.free_target_position is not None:
                draw_path(nav.img_map, nav.graph, nav.path_image_space, nav.stored_robot_position,
                          nav.free_robot_position,
                          nav.stored_target_position, nav.free_target_position)

    nav.map_rgba_f32[:, :, 2::-1] = nav.img_map / 255.0
    dpg.set_value('tag_map_texture', nav.map_rgba_f32.flatten())


def main():
    # BGR u8 frame retrieved from the camera
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    # RGBA f32 image for displaying with dearpygui
    frame_rgba_f32 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.float32)
    frame_rgba_f32[:, :, 3] = 1.0

    FRAME_ASPECT_RATIO = FRAME_WIDTH / FRAME_HEIGHT
    MAP_ASPECT_RATIO = MAP_WIDTH_PX / MAP_HEIGHT_PX

    # camera_matrix, distortion_coeffs = calibrate_camera(FRAME_WIDTH, FRAME_HEIGHT)
    # store_camera_to_json('./camera.json', camera_matrix, distortion_coeffs)
    # return
    camera_matrix, distortion_coeffs = load_camera_from_json('./camera.json')
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs,
                                                           (MAP_WIDTH_PX, MAP_HEIGHT_PX), 0,
                                                           (MAP_WIDTH_PX, MAP_HEIGHT_PX))

    video_thread = VideoThread(FRAME_WIDTH, FRAME_HEIGHT)

    client = ClientAsync()
    node = aw(client.wait_for_node())
    aw(node.lock())

    compile_ok = aw(compile_run_python_for_thymio(node))
    if not compile_ok:
        aw(node.unlock())

    aw(node.wait_for_variables())

    nav = Navigator(client, node)

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    nav.detector = detector

    timer = RepeatTimer(SAMPLING_TIME, run_navigation, args=[nav])
    timer.start()

    # Create interface
    dpg.create_context()
    build_interface(nav)

    # nav.start_time is just the reference for t=0 on the plots
    nav.start_time = time.time()

    while dpg.is_dearpygui_running():
        # Give the client the occasion to handle its work
        aw(client.sleep(0.005))

        # Get the latest frame
        is_frame_new = video_thread.get_frame(frame)

        if is_frame_new:
            # Correct camera distortions
            frame = cv2.undistort(frame, camera_matrix, distortion_coeffs, newCameraMatrix=new_camera_matrix)

            # Detect map corners
            corners, ids, rejected = detector.detectMarkers(frame)
            map_found, map_corners = detect_map(corners, ids)
            nav.map_found = map_found
            if map_found:
                # Correct perspective, and update the Navigator's frame
                matrix = get_perspective_transform(map_corners, MAP_WIDTH_PX, MAP_HEIGHT_PX)
                cv2.warpPerspective(frame, matrix, dsize=(MAP_WIDTH_PX, MAP_HEIGHT_PX), dst=nav.frame_map)

            # Draw a marker at each detected corner of the map
            for i in range(map_corners.shape[0]):
                cv2.drawMarker(frame, position=map_corners[i].astype(np.int32), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, thickness=2)

            # Convert BGR u8 to RGBA f32 for dearpygui
            frame_rgba_f32[:, :, 2::-1] = frame / 255.0
            dpg.set_value('tag_frame_texture', frame_rgba_f32.flatten())

        resize_image_to_fit_window('tag_camera_window', 'tag_frame_image', FRAME_ASPECT_RATIO)
        resize_image_to_fit_window('tag_map_window', 'tag_map_image', MAP_ASPECT_RATIO)

        dpg.render_dearpygui_frame()

    timer.cancel()

    aw(nav.node.stop())
    aw(nav.node.unlock())

    video_thread.stop()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
