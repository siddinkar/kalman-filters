import random
import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy import *

from ExtendedKalmanFilter import *


def unscented_transform(state, cov):
    L = len(state)
    chi = np.zeros((2 * L + 1, L))
    weights = np.zeros(2 * L + 1)

    chi[0] = state
    weights[0] = 0.25
    S = math.sqrt(cov)
    for i in range(1, 2 * L + 1):
        weights[i] = (1 - weights[0]) / (2 * L)
        if i <= L:
            chi[i] = state + (np.sqrt(L / (1 - weights[0])) * S)
        else:
            chi[i] = state - (np.sqrt(L / (1 - weights[0])) * S)

    return chi, weights


class UTC:
    x = np.zeros(2).T  # state vector (SOC_t, V_t) | always x_t|t
    u = np.zeros(1).T  # input vector current
    y = np.zeros(1).T  # measurement vector

    sigma_matrix = np.zeros((3, 2))
    y_matrix = np.zeros(3)
    U = np.zeros(3)
    Py = np.zeros((2, 2))  # Pyy
    Puy = np.zeros((2, 1))  # Pxy

    weights = np.zeros(2 * len(u) + 1)

    P = 0
    Q = 0
    K = np.zeros((2, 1))  # Kalman Gain


    def __init__(self, dt, SOC, tau, Cp, eta, r0, u, initial_sigma):
        self.dt = dt  # init
        self.x[0] = SOC
        self.F = np.array([  # init state transition matrix
            [1, 0],
            [0, np.exp(-dt / tau)]
        ])
        self.B = lambda x: np.array(
            [-eta * dt / Cp, R1(x * 100) * (1 - np.exp(-dt / tau))]
        )

        self.H = lambda x: np.array([OCV()(x * 100), -1]).reshape((1, 2))
        self.D = -r0

        self.P = initial_sigma
        self.Q = self.P * 0.01

        self.u = np.array([u])

    def predict(self):
        # self.u += np.random.normal(0, self.Q)
        self.U, self.weights = unscented_transform(self.u, self.P)

        y_pred = 0
        U_pred = 0
        P_pred = 0
        P_pred += self.Q
        self.Py = 0.00001
        self.Puy = 0
        for i in range(0, 2 * len(self.u) + 1):
            # Y_i
            self.sigma_matrix[i] = np.matmul(self.F, self.x) + self.B(self.x[0]) * self.U[i]
            self.y_matrix[i] = np.matmul(self.H(self.x[0]).reshape((2,)), self.sigma_matrix[i]) + self.D * self.U[i]

        for i in range(0, 2 * len(self.u) + 1):
            # Y_pred
            y_pred += self.weights[i] * self.y_matrix[i]
            # U_i
            U_pred += self.weights[i] * self.U[i]

        for i in range(0, 2 * len(self.u) + 1):
            r1 = self.U[i] - U_pred
            P_pred += self.weights[i] * np.matmul(r1, r1.T)
            # Pyy
            r2 = self.y_matrix[i] - y_pred
            self.Py += self.weights[i] * r2 * r2
            # Pxy
            self.Puy += self.weights[i] * r1 * r2

        return y_pred, U_pred, P_pred

    def update(self, y_ref):
        y_pred, U_pred, P_pred = self.predict()


        self.K = self.Puy / self.Py
        # print(self.K)

        U_correction = self.K * (y_ref - y_pred)
        P_correction = self.K * self.Py * self.K

        self.u = np.clip(U_pred + U_correction, -100, 100)
        self.P = P_pred - P_correction

        print(self.u)

        self.x = np.matmul(self.F, self.x) + self.B(self.x[0]) * self.u


class Environment:
    # ground truth and estimation states
    x_ref = np.zeros(2).T
    x_actual = np.zeros(2).T
    y_ref = 0
    y_actual = 0

    def __init__(self):
        # control signal pa rams
        self.freq = 1000  # samples
        self.total_time = 50  # seconds
        self.t = np.linspace(0, self.total_time, self.freq, False)
        self.current = np.cos(self.t/5.0) * np.exp(-self.t / 25.0) + 2.0
        init_pos_sigma = 1.0

        # System consts
        self.tau = 1
        self.Cp = 1620
        self.eta = 1.0
        self.r0 = .001
        SOC = 0.5
        init_u = 10.0

        self.x_ref[0] = SOC
        self.x_actual[0] = SOC

        self.dt = self.total_time / self.freq
        self.UTC = UTC(self.dt, SOC, self.tau, self.Cp, self.eta, self.r0, init_u, init_pos_sigma)

        self.y_actual = np.matmul(self.UTC.H(self.x_actual[0]).reshape((2,)), self.x_actual) + self.UTC.D * self.UTC.u

        self.y_ref_series = []
        self.x_act_series = []
        self.y_true_series = []
        self.error = []

    def run(self):
        for n in range(0, self.freq):

            self.x_ref = (np.matmul(self.UTC.F, self.x_ref)
                             + self.UTC.B(self.x_ref[0]) * self.UTC.u).reshape((2,))
            self.y_ref = self.current[n] #+ np.random.normal(0, 0.2)#np.matmul(self.UTC.H(self.x_ref[0]).reshape((2,)), self.x_ref) + self.UTC.D * u

            self.UTC.update(self.y_ref)
            self.x_actual = self.UTC.x
            self.x_act_series.append(self.x_actual[0])
            self.y_actual = np.matmul(self.UTC.H(self.x_actual[0]).reshape((2,)), self.x_actual) + self.UTC.D * self.UTC.u

            self.y_ref_series.append(self.y_ref)
            self.y_true_series.append(self.y_actual[0])
            self.error.append(self.y_ref - self.y_actual[0])

        self.y_ref_series = np.array(self.y_ref_series)
        self.y_true_series = np.array(self.y_true_series)
        self.x_act_series = np.array(self.x_act_series)
        self.error = np.array(self.error)


        title_font = {'fontname': 'Arial', 'size': '18', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'center'}
        axis_font = {'fontname': 'Arial', 'size': '16'}
        plt.plot(self.t, self.y_ref_series, label="Reference Signal")
        plt.plot(self.t, self.y_true_series, label="True Output")
        plt.plot(self.t, self.error, label="Error")
        plt.xlabel("t (s)", **axis_font)
        plt.title("Battery OCV", **title_font)
        plt.ylabel("Output Voltage (V_t)", **axis_font)
        plt.legend(loc='upper right')
        plt.savefig("fig3.png")
        plt.show()


if __name__ == "__main__":
    env = Environment()
    env.run()





