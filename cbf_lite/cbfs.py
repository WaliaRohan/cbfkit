import jax.numpy as jnp
from jax.scipy.special import erfinv

# CLF: V(x) = ||x - goal||^2
def vanilla_clf(x, goal):
    return jnp.linalg.norm(x - goal) ** 2

# CBF: h(x) = ||x - obstacle||^2 - safe_radius^2
def vanilla_cbf_circle(x, obstacle, safe_radius):
    return jnp.linalg.norm(x - obstacle) ** 2 - safe_radius**2

def vanilla_cbf_wall(x, wall_x):
    return wall_x - x[1]

### Belief CBF

import numpy as np


def belief_cbf_half_space(alpha, beta, delta, mu, sigma):
    '''
    alpha: linear gain from half-space constraint (alpha^T.x >= B, where x is the state)
    beta: constant from half-space constraint
    mu: mean of state belief
    sigma: covariance of state belief
    delta: probability of failure (we want system to be have probability of feailure less than delta)
    
    '''


    can't use this as is. need to find a way to only use x state

    term1 = jnp.dot(alpha.T, mu) - beta
    term2 = jnp.sqrt(2 * jnp.dot(alpha.T, jnp.dot(sigma, alpha))) * erfinv(1 - 2 * delta)
    return term1 - term2


    

