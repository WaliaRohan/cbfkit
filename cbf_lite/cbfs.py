import numpy as np


# CLF: V(x) = ||x - goal||^2
def vanilla_clf(x, goal):
    return np.linalg.norm(x - goal) ** 2

# CBF: h(x) = ||x - obstacle||^2 - safe_radius^2
def vanilla_cbf(x, obstacle, safe_radius):
    return np.linalg.norm(x - obstacle) ** 2 - safe_radius**2