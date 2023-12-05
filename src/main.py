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

    def __init__(self):
        self.start_time = time.time()
        self.node = None
        self.first_estimate = True
        # BGR u8 image destined to image processing
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
        self.sample_time_history = []
        self.estimated_pose_history = []
        self.measured_pose_history = []


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


def build_static_graph(nav: Navigator):
    obstacle_mask = get_obstacle_mask(nav.frame_map, nav.robot_position if nav.robot_found else None,
                                      nav.target_position if nav.target_found else None)
    # Note: the minimum distance to any obstacle is 'DILATION_SIZE_PX - approx_poly_epsilon'
    approx_poly_epsilon = 2
    nav.regions = extract_contours(obstacle_mask, approx_poly_epsilon)
    nav.graph = build_graph(nav.regions)


def callback_button_build_map(sender, app_data, user_data):
    assert isinstance(user_data, Navigator)
    build_static_graph(user_data)


# FIXME: temporary
def callback_update_graph(sender, app_data, user_data):
    assert isinstance(user_data, Navigator)
    nav = user_data
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


def resize_plot_to_fit_window(window: Union[int, str], plot: Union[int, str], image_aspect_ratio: float):
    window_width, window_height = dpg.get_item_rect_size(window)
    if window_width <= 0 or window_height <= 0:
        return

    # Use the item position within the window to deduce the window's border width and title bar height
    plot_pos_x, plot_pos_y = dpg.get_item_pos(plot)
    window_content_width = window_width - 2 * plot_pos_x
    window_content_height = window_height - plot_pos_y - plot_pos_x
    plot_width, plot_height = dpg.get_item_rect_size(plot)

    # TODO: it should be possible to correctly resize the plot, but need to make a drawing

    # dpg.set_axis_limits('tag_map_axis_x', 0, MAP_WIDTH_PX)
    # dpg.set_axis_limits('tag_map_axis_y', 0, MAP_HEIGHT_PX)
    # print(dpg.get_axis_limits('tag_map_axis_x'))
    # print(dpg.get_axis_limits('tag_map_axis_y'))


def update_plots(nav: Navigator):
    if len(nav.sample_time_history) < 2:
        return

    num_samples_to_plot = dpg.get_value('tag_samples_slider')
    num_samples_to_plot = min(num_samples_to_plot, len(nav.sample_time_history))
    time_data = np.array(nav.sample_time_history)[-num_samples_to_plot:].tolist()
    estimated_pose_arr = np.array(nav.estimated_pose_history)[-num_samples_to_plot:]
    measured_pose_arr = np.array(nav.estimated_pose_history)[-num_samples_to_plot:]
    estimated_theta_degrees = np.rad2deg(estimated_pose_arr[:, 2])[-num_samples_to_plot:]
    measured_theta_degrees = np.rad2deg(measured_pose_arr[:, 2])[-num_samples_to_plot:]

    dpg.set_value('tag_series_x_est', [time_data, estimated_pose_arr[:, 0].tolist()])
    dpg.set_value('tag_series_y_est', [time_data, estimated_pose_arr[:, 1].tolist()])
    dpg.set_value('tag_series_x_meas', [time_data, measured_pose_arr[:, 0].tolist()])
    dpg.set_value('tag_series_y_meas', [time_data, measured_pose_arr[:, 1].tolist()])
    dpg.set_value('tag_series_theta_est', [time_data, estimated_theta_degrees.tolist()])
    dpg.set_value('tag_series_theta_meas', [time_data, measured_theta_degrees.tolist()])

    # Set limits on the time axis (will automatically change the time axis of the other plot, since they are linked)
    dpg.set_axis_limits('tag_plot_xy_axis_x', time_data[0], time_data[-1])

    # Set limits on the y axes
    min_pos = min(np.min(estimated_pose_arr[:, :2]), np.min(measured_pose_arr[:, :2]))
    max_pos = max(np.max(estimated_pose_arr[:, :2]), np.max(measured_pose_arr[:, :2]))
    min_theta = min(np.min(estimated_theta_degrees), np.min(measured_theta_degrees))
    max_theta = max(np.max(estimated_theta_degrees), np.max(measured_theta_degrees))
    print(f'{min_pos}, {max_pos}, {min_theta}, {max_theta}')
    pos_range = max_pos - min_pos
    theta_range = max_theta - min_theta
    ymin_xy = min_pos - pos_range * 0.1 if pos_range > 0 else min_pos - 1
    ymax_xy = max_pos + pos_range * 0.1 if pos_range > 0 else max_pos + 1
    ymin_theta = min_theta - theta_range * 0.1 if theta_range > 0 else min_theta - 1
    ymax_theta = max_theta + theta_range * 0.1 if theta_range > 0 else max_theta + 1
    dpg.set_axis_limits('tag_plot_xy_axis_y', ymin_xy, ymax_xy)
    dpg.set_axis_limits('tag_plot_theta_axis_y', ymin_theta, ymax_theta)


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
        dpg.add_button(label='Update graph', callback=callback_update_graph, user_data=nav)
        with dpg.plot(label='Map', tag='tag_map_plot', equal_aspects=True):
            dpg.add_plot_axis(dpg.mvXAxis, label='X [mm]', tag='tag_map_axis_x')
            with dpg.plot_axis(dpg.mvYAxis, label='Y [mm]', tag='tag_map_axis_y'):
                dpg.add_image_series(texture_tag='tag_map_texture', tag='tag_map_image_series', bounds_min=(0, 0),
                                     bounds_max=(MAP_WIDTH_PX, MAP_HEIGHT_PX))

    with dpg.window(label='Plots', tag='tag_plot_window', no_close=True):
        dpg.add_slider_int(label='Samples to plot', tag='tag_samples_slider', default_value=300, min_value=10,
                           max_value=1200)
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

        nav.sample_time_history.append(time.time() - nav.start_time)
        nav.measured_pose_history.append(measurements.flatten())
        nav.estimated_pose_history.append(estimated_state)
        update_plots(nav)

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

            nav.prev_x_est = np.array(nav.prev_x_est)

            u_l = np.clip(int(input_left / MMS_PER_MOTOR_SPEED), -500, 500)
            u_r = np.clip(int(input_right / MMS_PER_MOTOR_SPEED), -500, 500)
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

    nav.map_rgba_f32[:, :, 2::-1] = nav.img_map / 255.0
    dpg.set_value('tag_map_texture', nav.map_rgba_f32.flatten())


def run_local_navigation(loc_nav: LocalNavigator):
    prox_horizontal = [0] * 5
    for i in range(5):
        prox_horizontal[i] = list(loc_nav.node.v.prox.horizontal)[i]
    obst = [prox_horizontal[0], prox_horizontal[1], prox_horizontal[2], prox_horizontal[3], prox_horizontal[4]]

    aw(avoid_obstacles(loc_nav.node, loc_nav.client, loc_nav.state, loc_nav.side, obst, loc_nav.HTobst,
                       loc_nav.LTobst, loc_nav.motor_speed, loc_nav.step_back_time, loc_nav.spped_gain,
                       loc_nav.rotation_time))


def main():
    # BGR u8 frame retrieved from the camera
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    # RGBA f32 image for displaying with dearpygui
    frame_rgba_f32 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.float32)
    frame_rgba_f32[:, :, 3] = 1.0

    FRAME_ASPECT_RATIO = FRAME_WIDTH / FRAME_HEIGHT
    MAP_ASPECT_RATIO = MAP_WIDTH_PX / MAP_HEIGHT_PX

    # camera_matrix, distortion_coeffs = calibrate_camera(frame_width, frame_height)
    # store_to_json('./camera.json', camera_matrix, distortion_coeffs)
    # return
    camera_matrix, distortion_coeffs = load_camera_from_json('./camera.json')
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs,
                                                           (MAP_WIDTH_PX, MAP_HEIGHT_PX), 0,
                                                           (MAP_WIDTH_PX, MAP_HEIGHT_PX))

    video_thread = VideoThread(FRAME_WIDTH, FRAME_HEIGHT)

    nav = Navigator()

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.extendDictionary(nMarkers=6, markerSize=6)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    nav.detector = detector

    client = ClientAsync()
    nav.node = aw(client.wait_for_node())
    aw(nav.node.lock())

    compile_ok = aw(compile_run_python_for_thymio(nav.node))
    if not compile_ok:
        aw(nav.node.unlock())

    aw(nav.node.wait_for_variables())

    timer = RepeatTimer(SAMPLING_TIME, run_navigation, args=[nav])
    timer.start()

    loc_nav = LocalNavigator(client, nav.node)
    local_nav_timer = RepeatTimer(0.1, run_local_navigation, args=[loc_nav])
    # local_nav_timer.start()

    dpg.create_context()
    build_interface(nav)

    nav.start_time = time.time()

    while dpg.is_dearpygui_running():
        # Let the client the occasion to handle its work
        aw(client.sleep(0.005))

        # Get the latest frame
        is_frame_new = video_thread.get_frame(frame)

        if is_frame_new:
            # Correct camera distortions
            frame = cv2.undistort(frame, camera_matrix, distortion_coeffs, newCameraMatrix=new_camera_matrix)

            # Detect map corners
            corners, ids, rejected = detector.detectMarkers(frame)
            map_found, map_corners = detect_map(corners, ids)
            if map_found:
                # Correct perspective
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
        resize_plot_to_fit_window('tag_map_window', 'tag_map_plot', MAP_ASPECT_RATIO)

        dpg.render_dearpygui_frame()

    timer.cancel()
    local_nav_timer.cancel()

    aw(nav.node.stop())
    aw(nav.node.unlock())

    video_thread.stop()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
