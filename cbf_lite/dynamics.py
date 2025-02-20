import jax.numpy as jnp


class SimpleDynamics:
    """Simple system dynamics: dx/dt = f(x) + g(x) u"""
    
    def __init__(self, Q=None):
        self.f_matrix = jnp.array([[0, 0], [0, 0]])  # No drift for now
        self.g_matrix = jnp.array([[1, 0], [0, 1]])  # Identity control matrix
        if Q is None:
            self.Q = jnp.eye(self.f_matrix.shape[1])*0 
        else:
            self.Q = Q

    def f(self, x):
        """Drift dynamics: f(x)"""
        return self.f_matrix @ x  # Linear drift (zero in this case)

    def g(self, x):
        """Control matrix: g(x)"""
        return self.g_matrix  # Constant control input mapping
    

class DubinsDynamics:
    """2D Dubins Car Model with constant velocity and control over heading rate."""

    def __init__(self):
        """Initialize Dubins Car dynamics."""
        pass  # No parameters needed for a basic Dubins model

    def f(self, x):
        """
        Compute the drift dynamics f(x).
        
        State x = [x_pos, y_pos, theta, v]
        """
        x_pos, y_pos, theta, v = x
        return jnp.array([
            v * jnp.cos(theta),  # x_dot
            v * jnp.sin(theta),  # y_dot
            0,                   # theta_dot (no drift)
            0                    # v_dot (velocity is constant)
        ])

    def g(self, x):
        """
        Compute the control matrix g(x).
        
        Control u = [heading rate omega]
        """
        return jnp.array([
            [0],  # No control influence on x
            [0],  # No control influence on y
            [1],  # Control directly affects theta (heading)
            [0]   # No control influence on velocity
        ])
