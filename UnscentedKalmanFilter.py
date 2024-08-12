import random

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy import signal

import ExtendedKalmanFilter
from ExtendedKalmanFilter import *


# SOC Estimation
class UnscentedKalmanFilter:
    x = np.zeros(2).T  # state vector (SOC_t, V_t) | always x_t|t
    u = np.zeros(1).T  # input vector current
    y = np.zeros(1).T  # measurement vector

    sigma_matrix = np.zeros((2, 5))
    y_matrix = np.zeros(5)
    S = 0  # Pyy
    T = np.zeros((2, 1))  # Pxy
    z = 0

    mean_weights = np.zeros(2 * len(x) + 1)
    cov_weights = np.zeros(2 * len(x) + 1)

    Q = np.zeros((2, 2))  # process noise
    R = np.zeros((1, 1))  # measurement noise

    P = np.zeros((2, 2))  # state covariance matrix | always P_t|t
    H = np.zeros((1, 2))  # measurement transition matrix
    K = np.zeros((2, 1))  # Kalman Gain

    def __init__(self, dt, SOC, tau, Cp, eta, r0, initial_sigma, w_sigma, v_sigma):
        self.dt = dt  # init
        self.x[0] = SOC
        self.F = np.array([  # init state transition matrix
            [1, 0],
            [0, np.exp(-dt / tau)]
        ])
        self.B = lambda x: np.array(
            [-eta * dt / Cp, R1(x * 100) * (1 - np.exp(-dt / tau))]
        )

        np.fill_diagonal(self.Q, w_sigma)  # init process noise covariance matrix
        self.Q[0][0] /= 10
        np.fill_diagonal(self.R, v_sigma)  # init measurement noise covariance matrix

        self.H = lambda x: np.array([OCV()(x * 100), -1]).reshape((1, 2))
        self.D = -r0

        np.fill_diagonal(self.P, initial_sigma)  # init state covariance matrix
        self.P[0][0] /= 10

    def predict(self, u):
        self.sigma_matrix, self.mean_weights, self.cov_weights = self.unscented_transform(self.x, self.P, 1, 2, 1)

        for i in range(0, 2 * len(self.x) + 1):
            self.sigma_matrix[i] = np.matmul(self.F, self.sigma_matrix[i]) + self.B(self.x[0]) * u

        x_pred = np.zeros(2).T
        for i in range(0, 2 * len(self.x) + 1):
            x_pred += self.mean_weights[i] * self.sigma_matrix[i]

        P_pred = np.zeros((2, 2))
        P_pred += self.Q
        for i in range(0, 2 * len(self.x) + 1):
            r = self.sigma_matrix[i] - x_pred
            P_pred += self.cov_weights[i] * np.matmul(r, r.T)
        return x_pred, P_pred

    def add_measurement(self, x, P_pred, u):
        v = np.random.normal(
            0,
            self.R
        )
        chi, wm, wc = self.unscented_transform(x, P_pred, 1, 2, 1)
        for i in range(0, 2 * len(self.x) + 1):
            self.y_matrix[i] = np.matmul(self.H(x[0]).reshape((2,)), chi[i]) + self.D * u + v
        self.z = self.y_matrix[0]
        self.y = 0
        for i in range(0, 2 * len(self.x) + 1):
            self.y += wm[i] * self.y_matrix[i]
        self.S = 0
        self.S += self.R
        for i in range(0, 2 * len(self.x) + 1):
            self.S += wc[i] * (self.y_matrix[i] - self.y) ** 2
        self.T = np.zeros((2, 1))
        for i in range(0, 2 * len(self.x) + 1):
            self.T += (wc[i] * (chi[i] - self.x) * (self.y_matrix[i] - self.y)).reshape((2, 1))

    def correct(self, x_pred, P_pred):
        self.K = np.matmul(self.T, np.linalg.inv(self.S))

        x_correction = (self.K.T * (self.z - self.y)).reshape((2,))
        P_correction = np.matmul(self.K * self.S, self.K.T)

        self.x = x_pred + x_correction
        self.P = P_pred - P_correction

    def unscented_transform(self, state, cov, a, b, k):
        L = len(state)
        chi = np.zeros((2 * L + 1, L))
        mean_weights = np.zeros(2 * L + 1)
        cov_weights = np.zeros(2 * L + 1)

        scaling_factor = a ** 2 * (L + k) - L

        chi[0] = state
        mean_weights[0] = scaling_factor / (L + scaling_factor)
        cov_weights[0] = mean_weights[0]
        U, S, _ = la.svd(cov)
        for i in range(1, 2 * L + 1):
            mean_weights[i] = 1 / (2 * (L + scaling_factor))
            cov_weights[i] = 1 / (2 * (L + scaling_factor))
            if i <= L:
                chi[i] = state + (np.sqrt(L + scaling_factor) * U * np.sqrt(S))[i - 1]
            else:
                chi[i] = state - (np.sqrt(L + scaling_factor) * U * np.sqrt(S))[i - L - 1]

        # print(chi)
        # print(mean_weights)
        # print(cov_weights)

        return chi, mean_weights, cov_weights

    def get_latest(self):
        return self.x

    def get_output(self):
        p = OCV()
        return p(self.x[0] * 100) - self.x[1] - self.u * self.D


class Environment:
    # ground truth and estimation states
    x_actual = np.zeros(2).T
    x_hat_UKF = np.zeros(2).T
    x_hat_EKF = np.zeros(2).T
    u = 0

    def __init__(self):
        # control signal pa rams
        self.freq = 100  # samples
        self.total_time = 50  # seconds
        self.t = np.linspace(0, self.total_time, self.freq, False)
        self.current = 1 * np.ones(self.t.shape)
        for i in range(1, 5):
            for _ in range(0, i * 2):
                rand = random.randrange(2, 98)
                self.current[rand + 2] = i
                self.current[rand + 1] = i
                self.current[rand] = i
                self.current[rand - 1] = i
                self.current[rand - 2] = i

        # Kalman Filter consts
        init_pos_sigma = .01
        process_sigma = .000025
        meas_sigma = .001

        # System consts
        self.tau = 1
        self.Cp = 150
        self.eta = 1.0
        self.r0 = .001
        SOC = 1.0

        self.dt = self.total_time / self.freq
        self.UKF = UnscentedKalmanFilter(
            self.dt, SOC, self.tau, self.Cp, self.eta,
            self.r0, init_pos_sigma, process_sigma, meas_sigma)
        self.EKF = ExtendedKalmanFilter(
            self.dt, SOC, self.tau, self.Cp, self.eta,
            self.r0, init_pos_sigma, process_sigma, meas_sigma)
        self.x_actual = self.UKF.get_latest()

        # output signals
        self.state_true = []
        self.state_est_UKF = []
        self.state_est_EKF = []
        self.output_true = []
        self.output_EKF = []
        self.output_UKF = []
        self.err_UKF = []
        self.err_EKF = []

    def run(self):
        for n in range(0, self.freq):
            u = self.current[n]
            process_noise = np.random.multivariate_normal(
                (0, 0),
                self.UKF.Q
            ).T

            self.x_actual = (np.matmul(self.UKF.F, self.x_actual)
                             + self.UKF.B(self.x_actual[0]) * u + process_noise).reshape((2,))
            self.x_actual[0] = min(1.0, max(0.01, self.x_actual[0]))

            x_pred, P_pred = self.UKF.predict(u)
            self.UKF.add_measurement(self.x_actual, P_pred, u)
            self.UKF.correct(x_pred, P_pred)
            self.x_hat_UKF = self.UKF.get_latest()

            x_pred, P_pred = self.EKF.predict(u)
            self.EKF.add_measurement(self.x_actual, u)
            self.EKF.correct(x_pred, P_pred, u)
            self.x_hat_EKF = self.EKF.get_latest()

            self.state_true.append(self.x_actual.T)
            self.state_est_UKF.append(self.x_hat_UKF.T)
            self.state_est_EKF.append(self.x_hat_EKF.T)

            p = OCV()
            out = p(self.x_actual[0] * 100) - self.x_actual[1] - u * self.r0
            self.output_true.append(out)
            self.output_UKF.append(self.UKF.get_output())
            self.output_EKF.append(self.EKF.get_output())

        self.state_true = np.array(self.state_true).T
        self.state_est_UKF = np.array(self.state_est_UKF).T
        self.state_est_EKF = np.array(self.state_est_EKF).T
        self.err_UKF = np.abs(self.state_est_UKF - self.state_true)
        self.err_EKF = np.abs(self.state_est_EKF - self.state_true)

        # SOC
        # plt.plot(self.t, self.state_true[0], label="true")
        # plt.plot(self.t, self.err_UKF[0], label="UKF")
        # plt.plot(self.t, self.err_EKF[0], label="EKF")
        # plt.title("Error")
        # plt.xlabel("t (s)")
        # plt.ylabel("SOC Estimation Error")
        ax1 = plt.subplot()

        title_font = {'fontname': 'Arial', 'size': '18', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'center'}  # Bottom vertical alignment for more space
        axis_font = {'fontname': 'Arial', 'size': '16'}

        # plt.plot(self.t, self.output_true, label="true")
        # plt.plot(self.t, self.output_UKF, label="UKF")
        plt.plot(self.t, self.current)
        plt.xlabel("t (s)", **axis_font)
        plt.title("Current", **title_font)
        plt.ylabel("I (A)", **axis_font)
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(18)
        plt.subplots_adjust(bottom=0.13)
        plt.subplots_adjust(right=0.95)
        plt.rcParams.update({'font.size': 16})

        # # U1
        # self.axis2.plot(self.t, self.state_true[1], label="true")
        # self.axis2.plot(self.t, self.state_est_UKF[1], label="UKF")
        # self.axis2.plot(self.t, self.state_est_EKF[1], label="EKF")
        # self.axis2.set_title('U1')
        #
        # # Ut
        # self.axis3.plot(self.t, self.output_true, label="true")
        # self.axis3.plot(self.t, self.output_UKF, label="UKF")
        # self.axis3.plot(self.t, self.output_EKF, label="EKF")
        # self.axis3.set_title('Ut')
        #
        # self.axis1.plot(self.t, self.err_UKF[0], label="UKF")
        # self.axis1.plot(self.t, self.err_EKF[0], label="EKF")
        # # self.axis1.plot(self.state_true[0], self.output_true, label="OCV-SOC")
        # self.axis1.set_title('Error')

        # plt.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    env = Environment()
    env.run()
