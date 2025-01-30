from typing import Optional, Union

import jax.numpy as jnp
from jax import Array, random


def perfect(_t: float, x: Array, **_kwargs) -> Array:
    """Perfect sensor -- returns exactly the state.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)

    Returns:
        x (Array): state vector

    """
    return x


def unbiased_gaussian_noise(
    t: float,
    x: Array,
    sigma: Optional[Union[Array, None]] = None,
    key: Optional[Union[random.PRNGKey, None]] = None,
) -> Array:
    """Senses the state subject to additive, unbiased (zero-mean), Gaussian
    noise.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)
        sigma (Array): measurement model covariance matrix

    Returns:
        y (Array): measurement of full state vector

    """
    if sigma is None:
        sigma = 0.1 * jnp.eye((len(x)))

    if key is None:
        key = random.PRNGKey(0)

    # Calculate the dimension of the random vector
    dim = sigma.shape[0]

    # Generate a zero-mean Gaussian random vector with unit variance
    # (take n_initial_meas measurements at t = 0)
    n_initial_meas = 10
    max_iter = n_initial_meas if t == 0 else 1
    normal_samples = jnp.zeros((max_iter, dim))
    for ii in range(max_iter):
        key, subkey = random.split(key)
        normal_samples = normal_samples.at[ii, :].set(random.normal(subkey, shape=(dim,)))

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    if jnp.trace(abs(sigma)) > 0:
        chol = jnp.linalg.cholesky(sigma)
    else:
        chol = jnp.zeros(sigma.shape)

    sampled_random_vector = jnp.mean(jnp.dot(chol, normal_samples.T), axis=1)

    return x + jnp.mean(sampled_random_vector)

# additive, unbiased guassian noise proportional to mean
def unbiased_gaussian_noise_sd(
    t: float,
    x: Array,
    sigma: Optional[Union[Array, None]] = None,
    key: Optional[Union[random.PRNGKey, None]] = None
) -> Array:
    """Senses the state subject to additive, unbiased (zero-mean), Gaussian
    noise.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)
        sigma (Array): measurement model covariance matrix
        dimension: Which dimensions has uncertainty

    Returns:
        y (Array): measurement of full state vector

    """

    dimension=1

    variance = 0.01*jnp.square(x[dimension-1])
    std_dev = jnp.sqrt(variance)

    if sigma is None:
        sigma = 0.1 * jnp.eye((len(x)))

    if key is None:
        key = random.PRNGKey(0)

    key = random.fold_in(key, t) # create a new key for each time step, based on original key

    # Calculate the dimension of the random vector
    dim = sigma.shape[0]

    # Generate a zero-mean Gaussian random vector with unit variance
    # (take n_initial_meas measurements at t = 0)
    n_initial_meas = 10
    max_iter = n_initial_meas if t == 0 else 1
    normal_samples = jnp.zeros((max_iter, dim))
    for ii in range(max_iter):
        key, subkey = random.split(key)
        normal_samples = normal_samples.at[ii, :].set(std_dev * random.normal(subkey, shape=(dim,)))

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    if jnp.trace(abs(sigma)) > 0:
        chol = jnp.linalg.cholesky(sigma)
    else:
        chol = jnp.zeros(sigma.shape)

    sampled_random_vector = jnp.mean(jnp.dot(chol, normal_samples.T), axis=1)

    new_x = x

    if not jnp.isnan(jnp.mean(sampled_random_vector)):
        new_x = x.at[dimension-1].set(x[dimension-1] + jnp.mean(sampled_random_vector))

    return new_x

def get_chol(sigma, dim):

    cov_matrix = sigma * jnp.eye(dim)

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    if jnp.trace(abs(cov_matrix)) > 0:
        chol = jnp.linalg.cholesky(cov_matrix)
    else:
        chol = jnp.zeros(cov_matrix.shape)

    return chol

def unbiased_gaussian_noise_mult(
    t: float,
    x: Array,
    sigma: Optional[Union[Array, None]] = None,
    key: Optional[Union[random.PRNGKey, None]] = None
) -> Array:
    """Senses the state subject to additive, unbiased (zero-mean), Gaussian
    noise.

    Args:
        t (float): time (sec)
        x (Array): state vector (ground truth)
        sigma (Array): measurement model covariance matrix
        dimension: Which dimensions has uncertainty

    Returns:
        y (Array): measurement of full state vector

    """
    # Multiplicative noise
    mu_u = 0.0174
    sigma_u = 10*2.916e-4 # 10 times more than what was shown in GEKF paper

    # Additive noise
    mu_v = -0.0386
    sigma_v = 7.997e-5

    if key is None:
        key = random.PRNGKey(0)

    key = random.fold_in(key, t) # create a new key for each time step, based on original key

    # Calculate the dimension of the random vector
    dim = len(x)

    # Generate a zero-mean Gaussian random vector with unit variance
    # (take n_initial_meas measurements at t = 0)
    n_initial_meas = 10
    max_iter = n_initial_meas if t == 0 else 1
    normal_samples = jnp.zeros((max_iter, dim))
    
    for ii in range(max_iter):
        key, subkey = random.split(key)
        normal_samples = normal_samples.at[ii, :].set(random.normal(subkey, shape=(dim,)))

    chol_u = get_chol(sigma_u, dim)
    chol_v = get_chol(sigma_v, dim)

    u_vector = 1 + mu_u + jnp.mean(jnp.dot(chol_u, normal_samples.T), axis=1)
    v_vector = mu_v + jnp.mean(jnp.dot(chol_v, normal_samples.T), axis=1)

    new_x = x + v_vector # add biased gaussian noise

    # Add multiplicative noise to second state
    state_idx = 1
    if not jnp.isnan(jnp.mean(u_vector)):
        new_x = x.at[state_idx].set(x[state_idx]*jnp.mean(u_vector))

    return new_x
