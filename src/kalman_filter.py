from typing import Optional

from parameters import *


def kalman_filter(measurements: Optional[np.ndarray], mu_km: np.ndarray, sig_km: np.ndarray,
                  u_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Global parameters
    d = 100
    H = np.eye(3)
    Q = np.array([[1.7, 0, 0], [0, 1.7, 0], [0, 0, 0.1]])
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.04]])

    # Prediction through the a priori estimate
    # estimated mean of the state
    mu_k_pred = np.array([[0.0], [0.0], [0.0]])
    mu_k_pred[0] = mu_km[0, 0] + (u_k[0] + u_k[1]) / 2 * SAMPLING_TIME * np.sin(
        mu_km[2, 0] + (u_k[0] - u_k[1]) / d * SAMPLING_TIME)
    mu_k_pred[1] = mu_km[1, 0] + (u_k[0] + u_k[1]) / 2 * SAMPLING_TIME * np.cos(
        mu_km[2, 0] + (u_k[0] - u_k[1]) / d * SAMPLING_TIME)
    mu_k_pred[2] = mu_km[2, 0] + (u_k[0] - u_k[1]) / d * SAMPLING_TIME

    # Jacobian of the motion model
    G_k = np.eye(3)
    G_k[0, 2] = (u_k[0] + u_k[1]) / 2 * np.cos(mu_km[2, 0] + 1 / 2 * (u_k[0] - u_k[1]) / d)
    G_k[1, 2] = -(u_k[0] + u_k[1]) / 2 * np.sin(mu_km[2, 0] + 1 / 2 * (u_k[0] - u_k[1]) / d)

    # Estimated covariance of the state
    sig_k_pred = np.dot(G_k, np.dot(sig_km, G_k.T))
    sig_k_pred += Q

    if measurements is not None:
        y = measurements
    else:
        # If no measurements we consider our measurements to be the same as our a priori estimate as to cancel out the
        # effect of innovation
        y = mu_k_pred

    # Innovation / measurement residual
    i = y - np.dot(H, mu_k_pred)
    # measurement prediction covariance
    S = np.dot(H, np.dot(sig_k_pred, H.T)) + R

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(sig_k_pred, np.dot(H.T, np.linalg.inv(S)))

    # A posteriori estimate
    x_est = mu_k_pred + np.dot(K, i)
    sig_est = sig_k_pred - np.dot(K, np.dot(H, sig_k_pred))

    return x_est, sig_est
