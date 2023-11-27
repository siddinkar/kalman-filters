import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from ExtendedKalmanFilter import ExtendedKalmanFilter

# SOC Estimation
class UnscentedKalmanFilter:
    x = np.zeros(2).T  # state vector (SOC_t, V_t) | always x_t|t
    u = np.zeros(1).T  # input vector current
    y = np.zeros(1).T  # measurement vector

    sigma_matrix = np.zeros((2, 5))
    y_matrix = np.zeros(5)
    S = 0  # Pyy
    T = np.zeros((2, 1))  # Pxy

    mean_weights = np.zeros(2 * len(x) + 1)
    cov_weights = np.zeros(2 * len(x) + 1)

    Q = np.zeros((2, 2))  # process noise
    R = np.zeros((1, 1))  # measurement noise

    P = np.zeros((2, 2))  # state covariance matrix | always P_t|t
    H = np.zeros((1, 2))  # measurement transition matrix
    K = np.zeros((2, 1))  # Kalman Gain

    def __init__(self, dt, tau, Cp, eta, r0, r1, initial_sigma, w_sigma, v_sigma):
        self.dt = dt  # init
        self.F = np.array([  # init state transition matrix
            [1, 0],
            [0, np.exp(-dt / tau)]
        ])
        self.B = np.array(  # init control input matrix
            [[-eta * dt / Cp], [r1 * (1 - np.exp(-dt / tau))]]
        )

        np.fill_diagonal(self.Q, w_sigma)  # init process noise covariance matrix
        np.fill_diagonal(self.R, v_sigma)  # init measurement noise covariance matrix

        self.H = np.array([1, -1]).reshape((1, 2))
        self.D = r0

        np.fill_diagonal(self.P, initial_sigma)  # init state covariance matrix
        self.x = np.random.multivariate_normal(  # random state with multivariate distribution P
            (1.0, 12.7),
            self.P
        ).T

    def predict(self, u):
        self.sigma_matrix, self.mean_weights, self.cov_weights = self.unscented_transform(self.x, self.P, 1, 0, 1)
        self.sigma_matrix = np.matmul(self.F, self.sigma_matrix) + np.matmul(self.B, np.full((1, 5), u))
        x_pred = np.zeros(2).T
        for i in range(0, 2 * len(self.x) + 1):
            x_pred += self.mean_weights[i] * self.sigma_matrix.T[i]

        P_pred = np.zeros((2, 2))
        P_pred += self.Q
        for i in range(0, 2 * len(self.x) + 1):
            r = self.sigma_matrix.T[i] - x_pred
            P_pred += self.cov_weights[i] * np.matmul(r, r.T)
        return x_pred, P_pred

    def add_measurement(self, x_measurement, P_pred):
        chi, wm, wc = self.unscented_transform(x_measurement, P_pred, 1, 0, 1)
        self.y_matrix = np.matmul(self.H, chi).reshape(5)
        self.y = 0
        for i in range(0, 2 * len(self.x) + 1):
            self.y += wm[i] * self.y_matrix[i]
        self.S = 0
        self.S += self.R
        for i in range(0, 2 * len(self.x) + 1):
            self.S += wc[i] * (self.y_matrix[i] - self.y) ** 2
        self.T = np.zeros((2, 1))
        for i in range(0, 2 * len(self.x) + 1):
            self.T += (wc[i] * (chi.T[i] - self.x) * (self.y_matrix[i] - self.y)).reshape((2, 1))

    def correct(self, x_pred, P_pred):
        self.K = np.matmul(self.T, np.linalg.inv(self.S))

        x_correction = np.matmul(self.K, self.y - (np.matmul(self.H, self.x)))
        P_correction = np.matmul(self.K * self.S, self.K.T)

        self.x = x_pred + x_correction
        self.P = P_pred - P_correction

    def unscented_transform(self, state, cov, a, b, k):
        L = len(state)
        chi = np.zeros((2 * L + 1, L))
        mean_weights = np.zeros(2 * L + 1)
        cov_weights = np.zeros(2 * L + 1)

        scaling_factor = a**2 * (L + k) - L

        chi[0] = state
        mean_weights[0] = scaling_factor / (L + scaling_factor)
        cov_weights[0] = mean_weights[0]
        for i in range(1, 2 * L + 1):
            mean_weights[i] = 1 / (2 * (L + scaling_factor))
            cov_weights[i] = 1 / (2 * (L + scaling_factor))
            cov = nearestPD((L + scaling_factor) * cov)
            if i <= L:
                chi[i] = state + np.linalg.cholesky(cov)[i - 1]
            else:
                chi[i] = state - np.linalg.cholesky(cov)[i - L - 1]

        return chi.T, mean_weights, cov_weights



    def get_latest(self):
        return self.x

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


class Environment():
    # ground truth and estimation states
    x_actual = np.zeros(2).T
    x_hat_UKF = np.zeros(2).T
    x_hat_EKF = np.zeros(2).T
    u = np.zeros(1).T

    fig, ((axis1, axis2), (axis3, axis4)) = plt.subplots(2, 2)

    def __init__(self):
        # control signal pa rams
        self.freq = 100  # samples
        self.total_time = 5  # seconds
        self.t = np.linspace(0, self.total_time, self.freq, False)
        self.current = np.zeros(self.t.shape)

        # Kalman Filter
        init_pos_sigma = .00001
        process_sigma = .05
        meas_sigma = .05
        self.dt = self.total_time / self.freq
        self.UKF = UnscentedKalmanFilter(self.dt, 1, 300, 1., 1, 1, init_pos_sigma, process_sigma, meas_sigma)
        self.EKF = ExtendedKalmanFilter(self.dt, 1, 300, 1., 1, 1, init_pos_sigma, process_sigma, meas_sigma)
        self.x_actual = self.UKF.get_latest()

        # output signals
        self.state_true = []
        self.state_est_UKF = []
        self.state_est_EKF = []

    def run(self):
        if self.x_actual[0] > 0.0:
            for n in range(0, self.freq):
                print(self.x_actual[0])
                u = self.current[n]
                process_noise = np.random.multivariate_normal(
                    (0, 0),
                    self.UKF.Q
                ).T

                self.x_actual = (np.matmul(self.UKF.F, self.x_actual) + self.UKF.B.T * u + process_noise).reshape((2,))
                x_pred, P_pred = self.UKF.predict(u)
                self.UKF.add_measurement(self.x_actual, P_pred)
                self.UKF.correct(x_pred, P_pred)
                self.x_hat_UKF = self.UKF.get_latest()

                x_pred, P_pred = self.EKF.predict(u)
                self.EKF.add_measurement(self.x_actual, u)
                self.EKF.correct(x_pred, P_pred)
                self.x_hat_EKF = self.EKF.get_latest()

                self.state_true.append(self.x_actual.T)
                self.state_est_UKF.append(self.x_hat_UKF.T)
                self.state_est_EKF.append(self.x_hat_EKF.T)

        else:
            pass


        self.state_true = np.array(self.state_true).T
        self.state_est_UKF = np.array(self.state_est_UKF).T
        self.state_est_EKF = np.array(self.state_est_EKF).T

        # UKF SOC
        self.axis1.plot(self.t, self.state_true[0], label="true")
        self.axis1.plot(self.t, self.state_est_UKF[0], label="estimation")
        self.axis1.set_title('SOC')

        # UKF TV
        self.axis2.plot(self.t, self.state_true[1], label="true")
        self.axis2.plot(self.t, self.state_est_UKF[1], label="estimation")
        self.axis2.set_title('Terminal Voltage')

        # EKF SOC
        self.axis3.plot(self.t, self.state_true[0], label="true")
        self.axis3.plot(self.t, self.state_est_EKF[0],  label="estimation")
        self.axis1.set_title('SOC')

        # EKF
        self.axis4.plot(self.t, self.state_true[1], label="true")
        self.axis4.plot(self.t, self.state_est_EKF[1], label="estimation")
        self.axis2.set_title('Terminal Voltage')

        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = Environment()
    env.run()


