from parameters import *


def astolfi_control(state, goal_state) -> tuple[float, float, bool]:
    """
    Returns the left desired speed, right desired speed and whether the goal was reached
    """

    Kp = 8  # > 0
    Ka = 75  # > Kp
    goal_radius = 20

    delta_x = goal_state[0] - state[0]
    delta_y = goal_state[1] - state[1]
    reference_angle = -np.arctan2(delta_x, delta_y)

    rho = np.sqrt(delta_x ** 2 + delta_y ** 2)
    alpha = reference_angle - state[2]
    if alpha < -np.pi:
        alpha = 2 * np.pi + alpha
    elif alpha > np.pi:
        alpha = - 2 * np.pi + alpha

    if rho <= goal_radius:
        return 0, 0, True

    power = 1/4
    K_power = 140 / (goal_radius**power)
    v = Kp * K_power * rho**power
    # v = Kp * rho
    omega = Ka * alpha

    u_r = (ROBOT_CENTER_TO_WHEEL * omega + v) / ROBOT_WHEEL_RADIUS
    u_l = (v - ROBOT_CENTER_TO_WHEEL * omega) / ROBOT_WHEEL_RADIUS

    speed_threshold = 80
    u_r = np.clip(u_r, -speed_threshold, speed_threshold)
    u_l = np.clip(u_l, -speed_threshold, speed_threshold)

    return u_l, u_r, False
