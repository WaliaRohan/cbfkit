import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

class NonlinearEstimator:
    """Extended Kalman Filter (EKF) for state estimation."""
    
    def __init__(self, dynamics, sensor_model, dt, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.sensor_model = sensor_model  # Sensor model
        self.dt = dt  # Time step

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(2) * 0.1  
        self.Q = Q if Q is not None else jnp.eye(2) * 0.01  # Process noise covariance
        self.R = R if R is not None else jnp.eye(2) * 0.05  # Measurement noise covariance

    def predict(self, x_hat, u):
        """Predict step of EKF."""
        # Nonlinear state propagation
        x_pred = x_hat + self.dt * (self.dynamics.f(x_hat) + self.dynamics.g(x_hat) @ u)

        # Compute Jacobian of dynamics (linearization)
        F_x = jax.jacobian(lambda x: x + self.dt * (self.dynamics.f(x) + self.dynamics.g(x) @ u))(x_hat)

        # Update covariance
        self.P = F_x @ self.P @ F_x.T + self.Q
        return x_pred

    def update(self, x_pred, z):
        """Measurement update step of EKF."""
        H_x = jnp.eye(2)  # Jacobian of measurement model (assuming direct state observation)
        y = z - self.sensor_model(x_pred)  # Innovation (difference between measured and predicted state)

        # Kalman gain
        S = H_x @ self.P @ H_x.T + self.R
        K = self.P @ H_x.T @ linalg.inv(S)

        # Update state estimate
        x_est = x_pred + K @ y

        # Update covariance
        self.P = (jnp.eye(2) - K @ H_x) @ self.P
        return x_est
