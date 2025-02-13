from functools import partial
import jax.numpy as jnp
from cbfs import belief_cbf_half_space as h_b

# Define fixed parameters
alpha = jnp.eye(4)  # Example 4x4 matrix
beta = jnp.ones((4, 1))  # Example 4x1 vector
delta = 0.05  # Probability threshold
# Partially apply h_b with fixed alpha, beta, and sigma
h_b_partial = partial(h_b, alpha, beta, delta)

# Example usage
mu = jnp.array([[0.5], [0.2], [-0.3], [0.1]])  # Example 4x1 mean vector

sigma = jnp.eye(4) * 0.1  # Example 4x4 covariance matrix

result = h_b_partial(mu, sigma)
print(result)
