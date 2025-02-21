from dynamics import SimpleDynamics
from cbfs import belief_cbf_half_space as h_b
import jax.numpy as jnp
import jax


# Multi-dimensional
n = 2
alpha = jnp.array([1.0, 0.0])  # Example matrix
beta = jnp.array([5.0])  # Example vector
delta = 0.05  # Probability threshold

mu = jnp.array([0.2, -0.3])
vec_sigma = jnp.array([0.1, 0.05, 0.2])  # Upper triangular part
b = jnp.concatenate([mu, vec_sigma])

output = h_b(b, alpha, beta, delta, 2)

# Compute gradient automatically
grad_h_b = jax.grad(h_b, argnums=0)(b, alpha, beta, delta, 2)