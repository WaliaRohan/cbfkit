import jax.numpy as jnp


# CLF: V(x) = ||x - goal||^2
def vanilla_clf(x, goal):
    return jnp.linalg.norm(x - goal) ** 2

# CBF: h(x) = ||x - obstacle||^2 - safe_radius^2
def vanilla_cbf(x, obstacle, safe_radius):
    return jnp.linalg.norm(x - obstacle) ** 2 - safe_radius**2

### Belief CBF

import numpy as np


def h_b(alpha, beta, mu, sigma, delta):
    '''
    alpha: linear gain from half-space constraint (alpha^T.x >= B, where x is the state)
    beta: constant from half-space constraint
    mu: mean of state belief
    sigma: covariance of state belief
    delta: probability of failure (we want system to be have probability of feailure less than delta)
    
    '''

    term1 = alpha.T @ mu - beta
    term2 = np.sqrt(2*alpha.T @ sigma @ alpha) * np.erfinv(1 - 2 * delta)
    return term1 - term2


    

