from parameters import *


def kalman_filter(measurements, mu_km, sig_km, u_k):
    # global parameters
    d = 100
    H = np.eye(3)
    r1 = 1.7
    r2 = 1.7
    r3 = 0.1
    Q = np.array([[r1, 0, 0], [0, r2, 0], [0, 0, r3]])
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
    sig_k_pred = sig_k_pred + Q if Q is not None else sig_k_pred

    if True:  # camera:
        y = measurements

    else:
        # if no measurements we consider our measurements to be the same as our a priori estimate as to cancel out the
        # effect of innovation
        y = mu_k_pred
        # print('y:',y)
        # print('--------')

    # innovation / measurement residual
    i = y - np.dot(H, mu_k_pred)
    # print('np.dot(H, mu_k_pred): ',np.dot(H, mu_k_pred))
    # print('innovation:', i)
    # print('--------')
    # measurement prediction covariance
    S = np.dot(H, np.dot(sig_k_pred, H.T)) + R

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(sig_k_pred, np.dot(H.T, np.linalg.inv(S)))

    # a posteriori estimate
    # print('mu_k_pred:', mu_k_pred)
    # print('--------')
    x_est = mu_k_pred + np.dot(K, i)
    # print('mu_k_pred + np.dot(K,i): ',mu_k_pred + np.dot(K,i))
    # print('--------')
    sig_est = sig_k_pred - np.dot(K, np.dot(H, sig_k_pred))

    return x_est, sig_est
