import random
import math
import numpy as np
import scipy.linalg
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy import *



m  = 1.     # mass [kg]
g  = 9.81    # gravity [m/s^2]
Ix = 0.2    # inertia about x [kg*m^2]
Iy = 0.2    # inertia about y [kg*m^2]
Iz = 0.3    # inertia about z [kg*m^2]

def f_of_x(x):

    # Unpack:
    X    = x[0]
    Y    = x[1]
    Z    = x[2]
    psi  = x[3]
    theta= x[4]
    phi  = x[5]
    Xdot = x[6]
    Ydot = x[7]
    Zdot = x[8]
    p    = x[9]
    q    = x[10]
    r    = x[11]

    cphi   = np.cos(phi)
    sphi   = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    ttheta = np.tan(theta)
    cpsi   = np.cos(psi)
    spsi   = np.sin(psi)

    # f(x) based on the given formula (matching your references):
    f = np.zeros(12)
    # Xdot, Ydot, Zdot
    f[0] = Xdot
    f[1] = Ydot
    f[2] = Zdot

    # psi̇, thetȧ, phi̇

    f[3] = q*sphi/ctheta + r*cphi/ctheta
    f[4] = q*cphi - r*sphi
    f[5] = p + q*sphi*ttheta + r*cphi*ttheta

    f[6] = 0.0
    f[7] = 0.0
    f[8] = 0.0

    #  ṗ, q̇, ṙ
    f[9]  = ((Iy - Iz)/Ix) * q * r
    f[10] = ((Iz - Ix)/Iy) * p * r
    f[11] = ((Ix - Iy)/Iz) * p * q

    return f

def g1_of_x(x):

    phi  = x[5]
    theta= x[4]
    psi  = x[3]

    cphi   = np.cos(phi)
    sphi   = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi   = np.cos(psi)
    spsi   = np.sin(psi)

    g1 = np.zeros(12)
    # Fill positions 6,7,8 (the accelerations in Ẍ, Ÿ, Z̈):
    g1[6] = -1.0/m * ( sphi*spsi + cphi*cpsi*stheta )
    g1[7] = -1.0/m * ( cpsi*sphi - cphi*spsi*stheta )
    g1[8] = -1.0/m * ( cphi*ctheta )
    return g1

def g2_of_x(x):

    g2 = np.zeros(12)
    g2[9] = 1.0 / Ix
    return g2

def g3_of_x(x):

    g3 = np.zeros(12)
    g3[10] = 1.0 / Iy
    return g3

def g4_of_x(x):

    g4 = np.zeros(12)
    g4[11] = 1.0 / Iz
    return g4

def quad_next_state(x, u, dt):

    u1, u2, u3, u4 = u
    dx = f_of_x(x) \
         + g1_of_x(x)*u1 \
         + g2_of_x(x)*u2 \
         + g3_of_x(x)*u3 \
         + g4_of_x(x)*u4
    return x + (dt * dx)

def unscented_transform(state, cov):

    L = len(state)
    chi = np.zeros((2 * L + 1, L))
    weights = np.zeros(2 * L + 1)

    chi[0] = state
    # one simplistic weighting choice
    weights[0] = 0.20

    # SVD to get sqrt of cov
    U, S, _ = la.svd(cov)
    sqrt_cov = U * np.sqrt(S)

    # For i in [1..L], we add + or - the columns
    for i in range(1, 2 * L + 1):
        weights[i] = (1 - weights[0]) / (2 * L)
        if i <= L:
            chi[i] = state + (np.sqrt(L / (1 - weights[0])) * sqrt_cov)[i - 1]
        else:
            chi[i] = state - (np.sqrt(L / (1 - weights[0])) * sqrt_cov)[i-L - 1]

    return chi, weights

class UTC:
    def __init__(self, dt, u_init, step, initial_sigma):


        self.dt   = dt
        self.step = step


        self.x = np.random.random(12).T

        # 4D input
        self.u = np.array(u_init, dtype=float)
        self.P = np.eye(4) * initial_sigma
        # Process noise Q
        self.Q = self.P * 0.0001
        # Kalman gain
        self.K = np.zeros((4, 12))
        self.sigma_matrix = np.zeros((2 * len(self.u) + 1, 12))

        self.Py = np.eye(12)
        self.Puy = np.zeros((4, 12))

    def predict(self):



        self.U, self.weights = unscented_transform(self.u, self.P)


        dt = self.dt
        for i in range(2 * len(self.u) + 1):
            temp = self.x.copy()
            # do 'step' sub-steps to simulate multiple dt’s, or just 1 step if step=1
            for s in range(self.step):
                temp = quad_next_state(temp, self.U[i], dt)
            self.sigma_matrix[i] = temp




        x_pred = np.zeros(12)
        U_pred = np.zeros(4)
        for i in range(2 * len(self.u) + 1):
            w_i = self.weights[i]
            x_pred += w_i * self.sigma_matrix[i]
            U_pred += w_i * self.U[i]

        P_pred = np.eye(4) * 0.0001
        self.Py = np.eye(12) * 0.0001
        self.Puy = np.zeros((4, 12))

        for i in range(2 * len(self.u) + 1):
            r1 = np.array(self.U[i] - U_pred).reshape((4, 1))
            P_pred += self.weights[i] * np.matmul(r1, r1.T)
            # Pyy
            r2 = np.array(self.sigma_matrix[i] - x_pred).reshape((12, 1))
            self.Py += self.weights[i] * np.matmul(r2, r2.T)
            # Pxy
            self.Puy += self.weights[i] * np.matmul(r1, r2.T)
            print(np.linalg.norm(r1))
            print(np.linalg.norm(r2))


        return x_pred, U_pred, P_pred

    def update(self, x_ref):



        x_pred, U_pred, P_pred = self.predict()

        self.K = np.matmul(self.Puy, np.linalg.inv(self.Py))


        U_correction = np.matmul(self.K, (x_ref - x_pred))


        P_correction = self.K @ self.Py @ self.K.T



        self.u = (U_pred + U_correction)
        self.P = P_pred - P_correction
        self.x = quad_next_state(self.x, self.u, self.dt)

###############################################################################
# Environment, main loop, etc.
###############################################################################
class Environment:
    def __init__(self):
        self.freq = 1000
        self.total_time = 20
        self.t = np.linspace(0, self.total_time, self.freq, endpoint=False)


        self.ref = np.array([
            np.cos(self.t/2.0)*10.0,
            np.sin(self.t/3.0)*4.0,
            0.5*np.ones_like(self.t),
            np.cos(self.t/2.0)*6.0,
            np.sin(self.t/3.0)*6.0,
            0.5*np.ones_like(self.t),
            np.cos(self.t / 2.0) * 12.0,
            np.sin(self.t / 3.0) * 14.0,
            0.5 * np.ones_like(self.t),
            np.cos(self.t / 2.0) * 5.0,
            np.sin(self.t / 3.0) * 23.0,
            0.5 * np.ones_like(self.t)
        ]).T

        self.dt = self.total_time / self.freq

        self.x_series = []
        self.error = []
        self.pmax_series = []
        self.ref_series = []

    def run(self):
        init_u = np.array([0.0, 0.2, 0.0, 0.2])  # thrust, tau_phi, tau_theta, tau_psi
        init_pos_sigma = 1.0
        step = 1

        # Create the UTC with 12D state
        self.UTC = UTC(self.dt, init_u, step, init_pos_sigma)

        for i in range(self.freq):
            # pick reference
            x_ref = self.ref[i]

            self.UTC.update(x_ref)

            # log
            self.x_series.append(np.linalg.norm(self.UTC.x))
            err = np.linalg.norm(self.UTC.x - x_ref)
            self.error.append(err)
            self.ref_series.append(np.linalg.norm(x_ref))

        self.x_series = np.array(self.x_series)
        self.error = np.array(self.error)
        self.ref_series = np.array(self.ref_series)


        # Plot
        # plt.figure()
        # plt.plot(self.t, self.error, label="Position Error Norm")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Error")
        # plt.title("Quadrotor Position Tracking Error")
        # plt.legend()
        # plt.show()

        plt.figure()
        plt.plot(self.t, self.x_series, label='True state')
        plt.plot(self.t, self.ref_series, '--', label='Ref state')
        plt.xlabel("Time (s)")
        plt.ylabel("State |x|")
        plt.title("Quadrotor States vs Reference")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    env = Environment()
    env.run()
