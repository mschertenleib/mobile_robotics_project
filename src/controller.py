import numpy as np
import math


def move_robot(r_speed, l_speed):
    return {
        "motor.left.target": [l_speed],
        "motor.right.target": [r_speed],
    }


def control(state, goal_state, switch, previous_angle_error, previous_dist_error, sampling_time):
    K_p_angle = 10;
    K_i_angle = 1;
    K_d_angle = 0.1;

    K_p_dist = 4;
    K_i_dist = 0.4;
    K_d_dist = 0.04;

    angle_error = goal_state[2]-state[2];
    angle_error_integral = previous_angle_error+angle_error*sampling_time;
    angle_error_derivative = (angle_error-previous_angle_error)/sampling_time;

    x_error_goal = abs(goal_state[0]-state[0]);
    y_error_goal = abs(goal_state[1]-state[1]);
    dist_error = math.sqrt(x_error_goal**2+y_error_goal**2);
    dist_error_integral = previous_dist_error+dist_error*sampling_time;
    dist_error_derivative = (dist_error-previous_dist_error)/sampling_time;

    robot_orientation = np.deg2rad(state[2]);
    goal_angle = np.deg2rad(goal_state[2]);

    angle_threshold = 0.5;
    linear_threshold = 10;

    if ((abs(angle_error)>angle_threshold) and (switch==0)):
        u_r = K_p_angle*angle_error+K_i_angle*angle_error_integral+K_d_angle*angle_error_derivative;
        u_l = -(K_p_angle*angle_error+K_i_angle*angle_error_integral+K_d_angle*angle_error_derivative);
        # print(angle_error, angle_error_integral, angle_error_derivative);
    else:
        switch = 1;
        u_r = 0;
        u_l = 0;

    if ((abs(dist_error_integral)>linear_threshold) and (switch==1)):
        # print(dist_error);
        u_r = K_p_dist*dist_error+K_i_dist*dist_error_integral+K_d_dist*dist_error_derivative;
        u_l = K_p_dist*dist_error+K_i_dist*dist_error_integral+K_d_dist*dist_error_derivative;
        # print(dist_error, dist_error_integral, dist_error_derivative);
    
    speed_threshold = 80;
    if (abs(u_r)>speed_threshold):
        if (u_r>0):
            u_r = speed_threshold;
        elif (u_r<0):
            u_r = -speed_threshold;
    if (abs(u_l)>speed_threshold):
        if (u_l>0):
            u_l = speed_threshold;
        elif (u_l<0):
            u_l = -speed_threshold;
            
    u_r = int(u_r/0.4348);
    u_l = int(u_l/0.4348);
    u = [u_r, u_l];

    return u, switch, angle_error, dist_error
