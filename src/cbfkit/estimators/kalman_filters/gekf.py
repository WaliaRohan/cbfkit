from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array

from cbfkit.integration import forward_euler as integrate
from cbfkit.utils.user_types import DynamicsCallable

global K_EKF
K_EKF = jnp.zeros((6, 6))


def get_global_k_ekf():
    global K_EKF
    return K_EKF


def set_global_k_ekf(k_mat):
    global K_EKF
    K_EKF = k_mat


def ct_gekf_dtmeas(
    Q: Array,
    R: Array,
    dynamics: DynamicsCallable,
    dfdx: Callable,
    h: Callable,
    dhdx: Callable,
    dt: float,
) -> Callable[[float, Array, Array, Array], Tuple[Array, Array]]:
    """Function defining the continuous-time EKF with discrete-time measurements.

    Arguments:
        Q (Array): process noise covariance matrix
        R (Array): measurement noise covariance matrix
        dynamics (DynamicsCallable): function handle to computing nonlinear dynamics
        h (Callable): measurement model
        dhdx (Callable): linearized measurement model

    Returns:
        step_ekf (Callable): function handle to compute the next EKF observer state

    """

    predict = predict_ct_dtmeas(Q, dynamics, dfdx, dt)
    update = update_dtmeas(R, h, dhdx)

    def step_ekf(
        t: float,
        y: Array,
        z: Optional[Union[Array, None]] = None,
        u: Optional[Union[Array, None]] = None,
        P: Optional[Union[Array, None]] = None,
    ) -> Tuple[Array, Array]:
        """Continuous-time implementation of Extended Kalman Filter (EKF) with
        discrete-time measurements.

        Arguments:
            t (float): time (in sec)
            y (Array): measurement
            z (Array or None): observer state
            u (Array or None): control input
            P (Array or None): Kalman covariance matrix

        Returns:
            z_new (Array): updated observer state
            P_new (Array): updated Kalman covariance matrix

        """
        if z is None and u is None and P is None:
            return initialize(y, R)

        z_predicted, P_predicted = predict(t, z, u, P)
        z_new, P_new = update(z_predicted, y, P_predicted)

        return z_new, P_new

    return step_ekf


#! Possibly implement this in a better fashion in the future
def initialize(y: Array, R: Array) -> Array:
    """Initialization for the continuous-time EKF with discrete-time measurements.

    Arguments:
        y (Array): measurement of state
        R (Array): measurement noise covariance matrix (proxy for initial covariance)


    Returns:
        z0 (Array): initial estimate of state
        P0 (Array): initial covariance of state estimate

    """
    return y, R


def predict_ct_dtmeas(
    Q: Array, dynamics: DynamicsCallable, dfdx: Callable, dt: float
) -> Callable[[float, Array, Array, Array], Tuple[Array, Array]]:
    """Function defining the prediction step for the continuous-time EKF with discrete-time measurements.

    Arguments:
        Q (Array): positive definite process noise covariance
        dynamics (DynamicsCallable): function handle to computing the nonlinear system dynamics
        dfdx (Callable): linearized dynamics model
        dt (float): timestep (sec)

    Returns:
        predict (Callable): function handle to compute EKF state and covariance matrix based on system model

    """

    def predict(t: float, z: Array, u: Array, P: Array) -> Tuple[Array, Array]:
        """Implementation of prediction step for the continuous-time EKF with discrete-time measurements.

        Arguments:
            t (float): time (sec)
            z (Array): observer state
            u (Array): control input
            P (Array): Kalman covariance matrix

        Returns:
            xk: predicted observer state
            Pk: predicted covariance matrix

        """
        # Compute xdot from system dynamics
        f, g = dynamics(z)
        zdot = f + jnp.matmul(g, u)
        zk = integrate(z, zdot, dt)

        # Compute Pdot from covariance dynamics
        Ff, Fg = dfdx(z) #  value of partial derivatives of f and g wrt x
        F = Ff + jnp.einsum("ijk,j->ik", Fg, u)
        # F = Ff + jnp.einsum('ijk,j->ki', Fg, u)
        Pdot = jnp.matmul(F, P) + jnp.matmul(P, F.T) + Q
        Pk = integrate(P, Pdot, dt)

        return zk, Pk

    return predict


def update_dtmeas(
    R: Array, h: Callable[[Array], Array], dhdx: Callable[[Array], Array]
) -> Callable[[float, Array, Array], Tuple[Array, Array]]:
    """Function defining the update step for (any) EKF with discrete-time measurements.

    Arguments:
        R (Array): measurement noise covariance matrix
        h (Callable): measurement model
        dhdx (Callable): linearized measurement model


    Returns:
        update (Callable): function handle to compute the updated EKF state and covariance matrix

    """

    def update(z: Array, y: Array, P: Array) -> Tuple[Array, Array]:
        """Update step for (any)) EKF with discrete-time measurements.

        Arguments:
            z (Array): predicted observer state
            y (Array): measurement
            P (Array): predicted Kalman covariance matrix

        Returns:
            x_new (Array): updated observer state
            P_new (Array): updated Kalman covariance matrix

        """

        # Multiplicative noise
        mu_u = 0.0174
        sigma_u = 10*2.916e-4 # 10 times more than what was shown in GEKF paper

        # Additive noise
        # mu_v = -0.0386
        mu_v = 0
        sigma_v = 7.997e-5

        H_dot = dhdx(z)

        h_z = h(z)  # Assuming h(z) is a 4x1 vector
        h_z = h_z.at[1].set(h_z[1] * (1 + mu_u))
        E = h_z + mu_v
        
        C = jnp.matmul(P, jnp.transpose(H_dot, axes=None))  # Perform the matrix multiplication
        C = C.at[1].set((1 + mu_u) * C[1])  # Modify only the second element
        
        M = jnp.diag(jnp.diag(jnp.matmul(H_dot, jnp.matmul(P, jnp.transpose(H_dot))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
        S = jnp.matmul(H_dot, jnp.matmul(P, jnp.transpose(H_dot, axes=None)))  # Perform matrix multiplication
        S = S.at[1].set(jnp.square(1 + mu_u) * S[1])  # Apply (1 + mu_u)^2 to the second element only
        M = M.at[1, 1].set(jnp.square(sigma_u) * M[1, 1])
        S = S + M + jnp.square(sigma_v)

        K = jnp.matmul(C, jnp.linalg.inv(S))
        z_new = z + jnp.matmul(K, y - E)
        P_new = P - jnp.matmul(K, jnp.transpose(C, axes=None))
        set_global_k_ekf(K)

        return z_new, P_new

    return update
