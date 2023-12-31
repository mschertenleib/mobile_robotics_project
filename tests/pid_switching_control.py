import numpy as np


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
    dist_error = np.sqrt(x_error_goal ** 2 + y_error_goal ** 2)
    dist_error_integral = previous_dist_error + dist_error * sampling_time
    dist_error_derivative = (dist_error - previous_dist_error) / sampling_time

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
    else:
        u_r = K_p_dist * dist_error + K_i_dist * dist_error_integral + K_d_dist * dist_error_derivative
        u_l = K_p_dist * dist_error + K_i_dist * dist_error_integral + K_d_dist * dist_error_derivative

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
