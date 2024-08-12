import random
import math
import numpy as np
import scipy.linalg
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
    U, S, _ = la.svd(cov)
    for i in range(1, 2 * L + 1):
        weights[i] = (1 - weights[0]) / (2 * L)
        if i <= L:
            chi[i] = state + (np.sqrt(L / (1 - weights[0])) * U * np.sqrt(S))[i-1]
        else:
            chi[i] = state - (np.sqrt(L / (1 - weights[0])) * U * np.sqrt(S))[i-L-1]

    return chi, weights


class UTC:
    x = np.zeros(3).T  # state vector (p, q, r) | always x_t|t
    u = np.zeros(4).T  # input vector

    sigma_matrix = np.zeros((9, 3))
    U = np.zeros((4, 3))
    Py = np.zeros((3, 3))  # Pyy
    Puy = np.zeros((4, 3))  # Puy

    weights = np.zeros(2 * len(u) + 1)

    P = np.zeros((4,4))
    Q = 0
    K = np.zeros((4, 3))  # Kalman Gain

    def __init__(self, dt, u, f_bar, initial_sigma):
        self.dt = dt  # init
        # Derivative discretization
        # self.A = np.array([  # discretized (A * dt + I)
        #     [-0.9967, 0 , 0.6176],
        #     [0, -0.5057, 0],
        #     [-0.0939, 0, -0.2127]
        # ]) * dt + np.eye(3)
        # print(self.A)
        self.f_bar = f_bar

        # Better Discretization
        self.A = np.array(scipy.linalg.expm(np.array([  # discretized (A * dt + I)
            [-0.9967, 0, 0.6176],
            [0, -0.5057, 0],
            [-0.0939, 0, -0.2127]
        ]) * dt))

        # Derivative discretization
        # self.B = np.array([
        #     [0, -4.2423, 4.2423, 1.4871],
        #     [1.6532, -1.2735, -1.2735, 0.0024],
        #     [0, -0.2805, 0.2805, -0.8823]
        # ]) * dt
        # print(self.B)


        # Better Discretization
        self.B = np.array([
            [0, -4.2423, 4.2423, 1.4871],
            [1.6532, -1.2735, -1.2735, 0.0024],
            [0, -0.2805, 0.2805, -0.8823]
        ])
        self.B = np.linalg.inv(self.A) @ (self.A - np.eye(3)) @ self.B

        self.x = np.array([200000,200000, 200000])

        self.f_xk = np.clip(np.random.rand(1000, 3) * f_bar, -f_bar, f_bar)

        first = (np.eye(3) - self.B @ self.K) @ self.A
        sec = (np.eye(3) - self.B @ self.K) @ self.B
        third = -self.K @ self.A
        fourth = (np.eye(4) - self.K @ self.B)
        self.Z = np.block([
            [first, sec],
            [third, fourth]
        ])

        self.P = np.eye(4) * initial_sigma
        self.Q = self.P * 0.01

        self.pmax = 0
        self.bound = 0

        self.u = np.array(u)

    def predict(self, n):
        #self.u += np.random.normal(0, self.Q)
        self.U, self.weights = unscented_transform(self.u, self.P)

        x_pred = np.zeros(3)
        U_pred = np.zeros(4)
        P_pred = np.eye(4) * self.Q
        self.Py = 0.001 * np.eye(3)
        self.Puy = np.zeros((4, 3))
        for i in range(0, 2 * len(self.u) + 1):
            # Y_i
            self.sigma_matrix[i] = np.matmul(self.A, self.x) + np.matmul(self.B, self.U[i])

        for i in range(0, 2 * len(self.u) + 1):
            # x_pred
            x_pred += self.weights[i] * self.sigma_matrix[i]
            # U_i
            U_pred += self.weights[i] * self.U[i]

        for i in range(0, 2 * len(self.u) + 1):
            r1 = np.array(self.U[i] - U_pred).reshape((4,1))
            P_pred += self.weights[i] * np.matmul(r1, r1.T)
            # Pyy
            r2 = np.array(self.sigma_matrix[i] - x_pred).reshape((3,1))
            self.Py += self.weights[i] * np.matmul(r2, r2.T)
            # Pxy
            self.Puy += self.weights[i] * np.matmul(r1, r2.T)

        return x_pred, U_pred, P_pred

    def update(self, x_ref, n):
        x_pred, U_pred, P_pred = self.predict(n)

        self.K = np.matmul(self.Puy, np.linalg.inv(self.Py))

        U_correction = np.matmul(self.K, (x_ref - x_pred))
        P_correction = self.K @ self.Py @ self.K.T

        self.u = self.clamp_input(U_pred + U_correction)
        self.P = P_pred - P_correction

        d = self.f_bar#np.array([np.sin(n/20.0) * self.f_bar, self.f_bar * np.cos(n / 10.0), self.f_bar * np.sin(n / 50.0)])
        self.x = np.matmul(self.A, self.x) + self.B @ self.u + self.f_xk[n]#np.array([np.sin(n/20.0) * self.f_bar]*3)#self.f_xk[n]
        self.Z = np.block([
            [(np.eye(3) - self.B @ self.K) @ self.A, (np.eye(3) - self.B @ self.K) @ self.B],
            [-self.K @ self.A, (np.eye(4) - self.K @ self.B)]
        ])
        p = scipy.linalg.solve_discrete_lyapunov(self.Z, np.eye(7))
        eig = np.linalg.eigvals(p)
        for i in range(len(eig)):
            eig[i] = eig[i].real
        self.pmax = max(eig)
        print(self.pmax)


    def clamp_input(self, u):
        uc = max(-(25/180) * math.pi, min((55/180) * math.pi, u[0]))
        ure = max(-(25 / 180) * math.pi, min((25 / 180) * math.pi, u[1]))
        ule = max(-(25 / 180) * math.pi, min((25 / 180) * math.pi, u[2]))
        ur = max(-(30 / 180) * math.pi, min((30 / 180) * math.pi, u[3]))
        return np.array([uc, ure, ule, ur])

class Environment:
    # ground truth and estimation states
    x_ref = np.zeros(2).T
    x_actual = np.zeros(2).T

    def __init__(self):
        # control signal pa rams
        self.freq = 1000  # samples
        self.total_time = 100  # seconds
        self.t = np.linspace(0, self.total_time, self.freq, False)
        self.ref = np.array([np.cos(self.t/5.0) * np.exp(-self.t / 25.0) * 2.0,
                    np.sin(self.t / 3.5) * np.exp(self.t/50.0),
                    np.cos(self.t / 1.5) * np.exp(-self.t / 25.0)]).T
        init_pos_sigma = 1.0

        # System consts
        #init_u = np.array([0.84, -0.1, 0.8, -0.2])
        init_u = np.array([0, 0, 0, 0])

        self.dt = self.total_time / self.freq
        self.UTC = UTC(self.dt, init_u, 1.57, init_pos_sigma)

        self.x_act_series = []
        self.error = []
        self.yk = []

        self.avg_error = []
        self.bound = []

    def run(self):
        init_u = np.array([0, 0, 0, 0])
        init_pos_sigma = 1.0
        self.UTC = UTC(self.dt, init_u, 0.5, init_pos_sigma)
        for n in range(0, self.freq):

            self.x_ref = self.ref[n]

            self.UTC.update(self.x_ref, n)
            self.x_actual = self.UTC.x
            R = self.UTC.f_bar * (np.linalg.norm(self.UTC.Z) * self.UTC.pmax + np.sqrt(np.linalg.norm(self.UTC.Z)**2 * self.UTC.pmax**2 + self.UTC.pmax))
            self.bound.append(R)
            self.x_act_series.append(self.x_actual)
            self.yk.append(np.linalg.norm(np.concatenate((self.x_actual, self.UTC.u))))
            self.error.append(np.linalg.norm(self.x_actual - self.ref[n], 2))


            #self.avg_error.append(np.average(self.error, axis=0, keepdims=True))
        # self.bound = [np.min(self.bound)] * 1000


        # self.x_act_series = np.array(self.x_act_series).T
        # self.error = np.array(self.error).T


        title_font = {'fontname': 'Arial', 'size': '18', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'center'}
        axis_font = {'fontname': 'Arial', 'size': '16'}
        #plt.plot(self.t, self.ref.T[0], label="Ref p")
        # plt.plot(self.t, self.ref.T[1], label="Ref q")
        # plt.plot(self.t, self.ref.T[2], label="Ref r")
        # plt.plot(self.t, self.x_act_series[0], label="True p")
        # plt.plot(self.t, self.x_act_series[1], label="True q")
        # plt.plot(self.t, self.x_act_series[2], label="True r")
        # plt.plot(np.linspace(0, 2.0, 50), self.avg_error, label="error")
        plt.plot(self.t, self.yk, label="y_k")
        # plt.plot(self.t, self.bound, label="bound")
        plt.xlabel("T(s)", **axis_font)
        plt.title("", **title_font)
        plt.ylabel("|y_k|", **axis_font)
        plt.legend(loc='upper right')
        plt.savefig("fig1.png")
        plt.show()


if __name__ == "__main__":
    env = Environment()
    env.run()





