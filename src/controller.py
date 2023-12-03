import math

import numpy as np


def set_robot_speed(left_speed, right_speed):
    return {
        "motor.left.target": [left_speed],
        "motor.right.target": [right_speed],
    }


def control(state, goal_state, switch, previous_angle_error, previous_dist_error, sampling_time):
    K_p_angle = 10
    K_i_angle = 1
    K_d_angle = 0.1

    K_p_dist = 4
    K_i_dist = 0.4
    K_d_dist = 0.04

    delta_x = goal_state[0] - state[0]
    delta_y = goal_state[1] - state[1]
    angle_target = -np.arctan2(delta_x, delta_y)
    angle_target = np.rad2deg(angle_target)
    angle_error = angle_target - state[2]
    angle_error_integral = previous_angle_error + angle_error * sampling_time
    angle_error_derivative = (angle_error - previous_angle_error) / sampling_time

    print(delta_x, delta_y, angle_target, state[0], state[1], state[2], switch)

    x_error_goal = abs(goal_state[0] - state[0])
    y_error_goal = abs(goal_state[1] - state[1])
    dist_error = math.sqrt(x_error_goal ** 2 + y_error_goal ** 2)
    dist_error_integral = previous_dist_error + dist_error * sampling_time
    dist_error_derivative = (dist_error - previous_dist_error) / sampling_time

    robot_orientation = np.deg2rad(state[2])
    goal_angle = np.deg2rad(angle_target)

    angle_threshold = 0.5
    linear_threshold = 10

    if abs(dist_error_integral) < linear_threshold:
        return [0, 0], switch, angle_error, dist_error

    if abs(angle_error) > angle_threshold:
        switch = 0
    else:
        switch = 1

    if switch == 0:
        u_r = K_p_angle * angle_error + K_i_angle * angle_error_integral + K_d_angle * angle_error_derivative
        u_l = -(K_p_angle * angle_error + K_i_angle * angle_error_integral + K_d_angle * angle_error_derivative)
        # print(angle_error, angle_error_integral, angle_error_derivative)
    else:
        # print(dist_error)
        u_r = K_p_dist * dist_error + K_i_dist * dist_error_integral + K_d_dist * dist_error_derivative
        u_l = K_p_dist * dist_error + K_i_dist * dist_error_integral + K_d_dist * dist_error_derivative
        # u_r += K_p_angle * angle_error + K_i_angle * angle_error_integral + K_d_angle * angle_error_derivative
        # u_l += -(K_p_angle * angle_error + K_i_angle * angle_error_integral + K_d_angle * angle_error_derivative)
        # print(dist_error, dist_error_integral, dist_error_derivative)

    speed_threshold = 40
    if abs(u_r) > speed_threshold:
        if u_r > 0:
            u_r = speed_threshold
        elif u_r < 0:
            u_r = -speed_threshold
    if abs(u_l) > speed_threshold:
        if u_l > 0:
            u_l = speed_threshold
        elif u_l < 0:
            u_l = -speed_threshold

    u = [u_r, u_l]

    return u, switch, angle_error, dist_error


def control2(state, goal_state, switch, previous_angle_error, previous_dist_error, sampling_time):
    K_p_angle = 10 / 10
    K_i_angle = 1
    K_d_angle = 0.1

    K_p_dist = 4 / 10
    K_i_dist = 0.4
    K_d_dist = 0.04

    delta_x = goal_state[0] - state[0]
    delta_y = goal_state[1] - state[1]
    angle_target = -np.arctan2(delta_x, delta_y)
    angle_target = np.rad2deg(angle_target)
    angle_error = angle_target - state[2]
    angle_error_integral = previous_angle_error + angle_error * sampling_time
    angle_error_derivative = (angle_error - previous_angle_error) / sampling_time

    print(delta_x, delta_y, angle_target, state[0], state[1], state[2], switch)

    x_error_goal = abs(goal_state[0] - state[0])
    y_error_goal = abs(goal_state[1] - state[1])
    dist_error = np.sqrt(x_error_goal ** 2 + y_error_goal ** 2)
    dist_error_integral = previous_dist_error + dist_error * sampling_time
    dist_error_derivative = (dist_error - previous_dist_error) / sampling_time

    angle_threshold = 5
    linear_threshold = 30

    if (abs(angle_error) > angle_threshold) and (switch == 0):
        u_r = K_p_angle * angle_error  # + K_i_angle * angle_error_integral + K_d_angle * angle_error_derivative
        u_l = -(K_p_angle * angle_error)  # + K_i_angle * angle_error_integral + K_d_angle * angle_error_derivative)
        # print(angle_error, angle_error_integral, angle_error_derivative)
    else:
        switch = 1
        u_r = 0
        u_l = 0

    if (abs(dist_error_integral) > linear_threshold) and (switch == 1) and (abs(angle_error) < angle_threshold):
        # print(dist_error)
        u_r = K_p_dist * dist_error  # + K_i_dist * dist_error_integral + K_d_dist * dist_error_derivative
        u_r += K_p_angle * angle_error
        u_l = K_p_dist * dist_error  # + K_i_dist * dist_error_integral + K_d_dist * dist_error_derivative
        u_l += -(K_p_angle * angle_error)
        # print(dist_error, dist_error_integral, dist_error_derivative)
    else:
        switch = 0
        u_r = 0
        u_l = 0

    speed_threshold = 80
    if abs(u_r) > speed_threshold:
        if u_r > 0:
            u_r = speed_threshold
        elif u_r < 0:
            u_r = -speed_threshold
    if abs(u_l) > speed_threshold:
        if u_l > 0:
            u_l = speed_threshold
        elif u_l < 0:
            u_l = -speed_threshold

    u = [u_r, u_l]

    return u, switch, angle_error, dist_error


def astolfi_control(state, goal_state):
    Kp = 4*2  # >0
    Ka = 25*4  # > kp
    Kb = -1e-8# <0
    l = 48;
    r = 22;

    delta_x = goal_state[0] - state[0];
    delta_y = goal_state[1] - state[1];
    reference_angle = -np.arctan2(delta_x, delta_y);

    rho = np.sqrt(delta_x ** 2 + delta_y ** 2)
    alpha = reference_angle - state[2]
    if alpha < -np.pi:
        alpha = 2 * np.pi + alpha
    elif alpha > np.pi:
        alpha = - 2 * np.pi + alpha
    beta = -reference_angle

    if rho < 20:
        return 0, 0, True

    v = Kp * rho;
    omega = Ka * alpha + Kb * beta;

    u_r = (l * omega + v) / r;
    u_l = (v - l * omega) / r;
    print(f'{delta_x = }, {delta_y = }, {alpha = }, {u_l = }, {u_r = }')

    speed_threshold = 80;
    if (abs(u_r) > speed_threshold):
        if (u_r > 0):
            u_r = speed_threshold;
        elif (u_r < 0):
            u_r = -speed_threshold;
    if (abs(u_l) > speed_threshold):
        if (u_l > 0):
            u_l = speed_threshold;
        elif (u_l < 0):
            u_l = -speed_threshold;

    return u_l, u_r, False
