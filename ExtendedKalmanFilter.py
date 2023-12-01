import numpy as np
import matplotlib.pyplot as plt


class ExtendedKalmanFilter:
    x = np.zeros(2).T  # state vector (SOC_t, V_t) | always x_t|t
    u = np.zeros(1).T  # input vector current
    z = np.zeros(1).T  # measurement vector

    Q = np.zeros((2, 2))  # process noise
    R = np.zeros((1, 1))  # measurement noise

    P = np.zeros((2, 2))  # state covariance matrix | always P_t|t
    C = np.zeros((1, 2))  # measurement transition matrix
    D = 0
    K = np.zeros((2, 1))  # Kalman Gain

    def __init__(self, dt, SOC, tau, Cp, eta, r0, initial_sigma, w_sigma, v_sigma):
        self.dt = dt  # init
        self.x[0] = SOC
        self.A = np.array([  # init state transition matrix
            [1, 0],
            [0, np.exp(-dt / tau)]
        ])
        self.B = np.array(  # init control input matrix
            [-eta * dt / Cp, 0.01 * (1 - np.exp(-dt / tau))]
        )

        self.C = lambda x: np.array([OCV().deriv()(x * 100), -1]).reshape((1, 2))
        self.D = -r0

        np.fill_diagonal(self.Q, w_sigma)  # init process noise covariance matrix
        self.Q[0][0] /= 10
        np.fill_diagonal(self.R, v_sigma)  # init measurement noise covariance matrix

        np.fill_diagonal(self.P, initial_sigma)  # init state covariance matrix
        self.P[0][0] /= 10

    def predict(self, u):
        x_pred = (np.matmul(self.A, self.x) + self.B * u).reshape((2,))
        P_pred = np.add(np.matmul(np.matmul(self.A, self.P), self.A.T), self.Q)
        return x_pred, P_pred

    def add_measurement(self, x, u):
        v = np.random.normal(
            0,
            self.R
        )
        self.z = np.matmul(self.C(x[0]), x) + self.D * u + v

    def correct(self, x_pred, P_pred, u):
        invertible = np.add(np.matmul(np.matmul(self.C(self.x[0]), P_pred), self.C(self.x[0]).T), self.R)
        self.K = np.matmul(np.matmul(P_pred, self.C(self.x[0]).T), np.linalg.inv(invertible))

        x_correction = np.matmul(self.K, self.z - (np.matmul(self.C(self.x[0]), x_pred) + self.D * u)).reshape((2,))
        P_correction = np.matmul(np.matmul(self.K, self.C(self.x[0])), P_pred)

        self.x = x_pred + x_correction
        self.P = P_pred - P_correction

    def get_latest(self):
        return self.x

    def get_output(self):
        p = OCV()
        return p(self.x[0] * 100) - self.x[1] - self.u * self.D


def OCV():
    b = [
        3.82e-10,  # b5
        -1.21e-7,  # b4
        1.51e-5,  # b3
        -9.3e-4,  # b2
        0.0295,  # b1
        2.85  # b0
        # -339774.4007,  # b6
        # 466661.2327,  # b7
        # -432425.3077,  # b8
        # 258290.4265,  # b9
        # -89772.06536,  # b10
        # 13792.76136  # b11
    ]
    return np.poly1d(b)


def R1(x):
    if x != 0:
        return 0.092 * x ** -0.8
    return 0
