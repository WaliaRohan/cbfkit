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
    term1 = jnp.dot(alpha.T, mu) - beta
    term2 = jnp.sqrt(2 * jnp.dot(alpha.T, jnp.dot(sigma, alpha))) * erfinv(1 - 2 * delta)
    return term1 - term2


def h_b_rb2(alpha, beta, mu, sigma, delta):
    '''
    Function for calculating barrier function for h_b where relative degree is 2
    
    '''

    roots = jnp.array([-0.1]) # Manually select root to be in left half plane
    polynomial = np.poly1d(roots, r=True)
    coeff = jnp.array(polynomial.coeffs)

    h_0 = h_b(alpha, beta, mu, sigma, delta)
    h_1 = jnp.grad(h_b, argnums=2) # 1st order derivative, take derivative with respect to mean of belief

    h_2 = jnp.array(h_0*coeff[0]) + jnp.array(h_1*coeff[1]) # equation 4 (belief paper), equation 38 (ECBF paper)

    return h_2



    

