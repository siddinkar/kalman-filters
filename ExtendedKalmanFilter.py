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

    def __init__(self, dt, tau, Cp, eta, r0, r1, initial_sigma, w_sigma, v_sigma):
        self.dt = dt  # init
        self.A = np.array([  # init state transition matrix
            [1, 0],
            [0, np.exp(-dt / tau)]
        ])
        self.B = np.array(  # init control input matrix
            [[-eta * dt / Cp], [r1 * (1 - np.exp(-dt / tau))]]
        )

        self.C = np.array([1, -1]).reshape((1, 2))
        self.D = r0

        np.fill_diagonal(self.Q, w_sigma)  # init process noise covariance matrix
        np.fill_diagonal(self.R, v_sigma)  # init measurement noise covariance matrix

        np.fill_diagonal(self.P, initial_sigma)  # init state covariance matrix
        self.P[0][0] /= 10
        self.x = np.random.multivariate_normal(  # random state with multivariate distribution P
            (1.0, 12.7),
            self.P
        ).T

    def predict(self, u):
        x_pred = (np.matmul(self.A, self.x) + self.B.T * u).reshape((2,))
        P_pred = np.add(np.matmul(np.matmul(self.A, self.P), self.A.T), self.Q)
        return x_pred, P_pred

    def add_measurement(self, x, u):
        v = np.random.normal(
            0,
            self.R
        ).T
        self.C[0][0] = OCV(x[0])
        self.z = np.matmul(self.C, x) + v

    def correct(self, x_pred, P_pred):
        invertible = np.add(np.matmul(np.matmul(self.C, P_pred), self.C.T), self.R)
        self.K = np.matmul(np.matmul(P_pred, self.C.T), np.linalg.inv(invertible))

        x_correction = np.matmul(self.K, self.z - np.matmul(self.C, x_pred)).reshape((2,))
        P_correction = np.matmul(np.matmul(self.K, self.C), P_pred)

        self.x = x_pred + x_correction
        self.P = P_pred - P_correction

    def get_latest(self):
        return self.x


def OCV(soc):
    b = [
        156.7270186,
        -1908.862407,
        13222.57837,
        -57986.229,
        169749.2076,
        -339774.4007,
        466661.2327,
        -432425.3077,
        258290.4265,
        -89772.06536,
        13792.76136
    ]
    sum = 0
    for i in range(len(b)):
        sum += (i + 1) * b[i] * soc ** i
    return sum



