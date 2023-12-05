from typing import Optional

from parameters import *


def kalman_filter(measurements: Optional[np.ndarray], mu_km: np.ndarray, sig_km: np.ndarray,
                  u_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Since our discretized motion model does not perfectly match reality, try to correct it by some extent.
    # Since forward Euler integration tends to extrapolate "too far" for smooth trajectories, artificially reduce the
    # contribution of the derivative to the state update. Note that this is just a "hack" that seems to work well enough
    # for this system, and the specific multiplication factor was found through experimentation. For a more precise
    # motion model, we should use a better integration scheme (Runge-Kutta 4 for example), and move the Kalman filter
    # computation to the hardware, to get significant lower latencies on the inputs, as well as using a higher sampling
    # frequency.
    EFFECTIVE_SAMPLING_TIME = SAMPLING_TIME * 0.8

    # Prediction through the a priori estimate
    # Estimated mean of the state
    mu_k_pred = np.array([[0.0], [0.0], [0.0]])
    mu_k_pred[0] = mu_km[0, 0] + (u_k[0] + u_k[1]) / 2 * EFFECTIVE_SAMPLING_TIME * -np.sin(
        mu_km[2, 0] - (u_k[0] - u_k[1]) / ROBOT_WHEEL_SPACING * EFFECTIVE_SAMPLING_TIME)
    mu_k_pred[1] = mu_km[1, 0] + (u_k[0] + u_k[1]) / 2 * EFFECTIVE_SAMPLING_TIME * np.cos(
        mu_km[2, 0] - (u_k[0] - u_k[1]) / ROBOT_WHEEL_SPACING * EFFECTIVE_SAMPLING_TIME)
    mu_k_pred[2] = mu_km[2, 0] - (u_k[0] - u_k[1]) / ROBOT_WHEEL_SPACING * EFFECTIVE_SAMPLING_TIME

    # Jacobian of the motion model
    G_k = np.eye(3)
    G_k[0, 2] = -(u_k[0] + u_k[1]) / 2 * np.cos(
        mu_km[2, 0] - (u_k[0] - u_k[1]) / ROBOT_WHEEL_SPACING * EFFECTIVE_SAMPLING_TIME)
    G_k[1, 2] = -(u_k[0] + u_k[1]) / 2 * np.sin(
        mu_km[2, 0] - (u_k[0] - u_k[1]) / ROBOT_WHEEL_SPACING * EFFECTIVE_SAMPLING_TIME)

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
    # Measurement prediction covariance
    S = sig_k_pred + KALMAN_R

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = sig_k_pred @ np.linalg.inv(S)

    # A posteriori estimate
    x_est = mu_k_pred + K @ i
    sig_est = sig_k_pred - K @ sig_k_pred

    return x_est, sig_est
