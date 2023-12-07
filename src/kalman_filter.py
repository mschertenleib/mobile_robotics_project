from typing import Optional

from parameters import *


def kalman_filter(measurements: Optional[np.ndarray], mu_km: np.ndarray, sig_km: np.ndarray, speed_left: float,
                  speed_right: float) -> tuple[np.ndarray, np.ndarray]:
    # Since our discretized motion model does not perfectly match reality, try to correct it by some extent.
    # Since forward Euler integration tends to extrapolate "too far" for smooth trajectories, artificially reduce the
    # contribution of the derivative to the state update. Note that this is just a "hack" that seems to work well enough
    # for this system, and the specific multiplication factor was found through experimentation. For a more precise
    # motion model, we should use a better integration scheme (Runge-Kutta 4 for example), and move the Kalman filter
    # computation to the hardware, to get significant lower latencies on the inputs, as well as using a higher sampling
    # frequency.
    EFFECTIVE_SAMPLING_TIME = SAMPLING_TIME * 0.8

    tangential_speed = (speed_left + speed_right) / 2
    angular_speed = (speed_right - speed_left) / ROBOT_WHEEL_SPACING
    delta_angle = angular_speed * EFFECTIVE_SAMPLING_TIME
    # Prediction through the a priori estimate
    x_est = mu_km[0]
    y_est = mu_km[1]
    angle_est = mu_km[2]
    # Estimated mean of the state
    mu_k_pred = np.zeros(3)
    mu_k_pred[0] = x_est + tangential_speed * EFFECTIVE_SAMPLING_TIME * -np.sin(angle_est + delta_angle)
    mu_k_pred[1] = y_est + tangential_speed * EFFECTIVE_SAMPLING_TIME * np.cos(angle_est + delta_angle)
    mu_k_pred[2] = angle_est + delta_angle

    # Jacobian of the motion model
    G_k = np.eye(3)
    G_k[0, 2] = -tangential_speed * np.cos(angle_est + delta_angle)
    G_k[1, 2] = -tangential_speed * np.sin(angle_est + delta_angle)

    # Estimated covariance of the state
    sig_k_pred = G_k @ sig_km @ G_k.T
    sig_k_pred += KALMAN_Q

    if measurements is not None:
        y = measurements
    else:
        # If no measurements we consider our measurements to be the same as our a priori estimate as to cancel out the
        # effect of innovation
        y = mu_k_pred

    # Innovation / measurement residual
    i = y - mu_k_pred

    # Correctly handle the case where the angle difference is discontinuous
    if i[2] < -np.pi:
        i[2] = 2 * np.pi + i[2]
    elif i[2] > np.pi:
        i[2] = - 2 * np.pi + i[2]

    # Measurement prediction covariance
    S = sig_k_pred + KALMAN_R

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = sig_k_pred @ np.linalg.inv(S)

    # A posteriori estimate
    x_est = mu_k_pred + K @ i
    sig_est = sig_k_pred - K @ sig_k_pred

    return x_est, sig_est
