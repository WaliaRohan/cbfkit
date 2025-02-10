import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
from cbfs import vanilla_cbf as cbf
from cbfs import vanilla_clf as clf
from dynamics import SimpleDynamics
from estimators import NonlinearEstimator as EKF
from jax import grad, jit

# from osqp import OSQP
from jaxopt import BoxOSQP as OSQP
from sensor import noisy_sensor as sensor

# Define simulation parameters
dt = 0.1  # Time step
T = 100 # Number of steps
u_max = 1.0

# Initial state (truth)
x_true = jnp.array([-1.0, -1.0])  # Start position
goal = jnp.array([2.0, 2.0])  # Goal position
obstacle = jnp.array([1.0, 1.0])  # Obstacle position
safe_radius = 0.5  # Safety radius around the obstacle

dynamics = SimpleDynamics()
estimator = EKF(dynamics, sensor, dt, x_init=x_true)

# Autodiff: Compute Gradients for CLF and CBF
grad_V = grad(clf, argnums=0)  # ∇V(x)
grad_h = grad(cbf, argnums=0)  # ∇h(x)

# OSQP solver instance
solver = OSQP()

@jit
def solve_qp(x_estimated):
    """Solve the CLF-CBF-QP using JAX & OSQP"""
    # Compute CLF components
    V = clf(x_estimated, goal)
    grad_V_x = grad_V(x_estimated, goal)  # ∇V(x)

    L_f_V = jnp.dot(grad_V_x, dynamics.f(x_estimated))
    L_g_V = jnp.dot(grad_V_x, dynamics.g(x_estimated))
    gamma = 1.0  # CLF gain

    # Compute CBF components
    h = cbf(x_estimated, obstacle, safe_radius)
    grad_h_x = grad_h(x_estimated, obstacle, safe_radius)  # ∇h(x)

    L_f_h = jnp.dot(grad_h_x, dynamics.f(x_estimated))
    L_g_h = jnp.dot(grad_h_x, dynamics.g(x_estimated))
    alpha = 1.0  # CBF gain

    # Define QP matrices
    Q = jnp.eye(2)  # Minimize ||u||^2
    c = jnp.zeros(2)  # No linear cost term

    A = jnp.vstack([
        L_g_V,   # CLF constraint
        -L_g_h,   # CBF constraint (negated for inequality direction)
        jnp.eye(2)
    ])

    u = jnp.hstack([
        -L_f_V - gamma * V,   # CLF constraint
        L_f_h + alpha * h,     # CBF constraint
        jnp.inf,
        u_max 
    ])

    l = jnp.hstack([
        -jnp.inf,
        -jnp.inf,
        -jnp.inf,
        -u_max
    ])

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    return sol

x_traj = []  # Store trajectory
x_meas = [] # Measurements
x_est = [] # Estimates
u_traj = []  # Store controls

x_traj.append(x_true)
x_estimated = estimator.get_belief()

# Simulation loop
for _ in range(T):
    # Solve QP
    u_opt = solve_qp(x_estimated).primal[0]

    # Apply control to the true state (x_true)
    x_true = x_true + dt * (dynamics.f(x_true) + dynamics.g(x_true) @ u_opt)

    # obtain current measurement
    x_measured =  sensor(x_true)

    # updated estimate 
    estimator.predict(u_opt)
    estimator.update(x_measured)
    x_estimated = estimator.get_belief()

    # Store for plotting
    x_traj.append(x_true.copy())
    u_traj.append(u_opt)
    x_meas.append(x_measured)
    x_est.append(x_estimated)
    # print(x_true)

# Convert to JAX arrays
x_traj = jnp.array(x_traj)

# Conver to numpy arrays for plotting
x_traj = np.array(x_traj)
x_meas = np.array(x_meas)
x_est = np.array(x_est)

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
plt.scatter(goal[0], goal[1], c="g", marker="*", s=200, label="Goal")
plt.scatter(obstacle[0], obstacle[1], c="r", marker="o", s=200, label="Obstacle")
circle = plt.Circle(obstacle, safe_radius, color="r", fill=False, linestyle="--")
plt.gca().add_patch(circle)
plt.xlim(-2, 3)
plt.ylim(-2, 3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("CLF-CBF QP-Controlled Trajectory (Autodiff with JAX)")
plt.legend()
plt.grid()
plt.show()

# Second figure: X component comparison
plt.figure(figsize=(6, 4))
plt.plot(x_meas[:, 0], color="green", label="Measured x", linestyle="dashed")
plt.plot(x_est[:, 0], color="orange", label="Estimated x", linestyle="dotted")
plt.plot(x_traj[:, 0], color="blue", label="True x")
plt.xlabel("Time step")
plt.ylabel("X component")
plt.legend()
plt.title("X Component Comparison")
plt.show()

# Third figure: Y component comparison
plt.figure(figsize=(6, 4))
plt.plot(x_meas[:, 1], color="green", label="Measured y", linestyle="dashed")
plt.plot(x_est[:, 1], color="orange", label="Estimated y", linestyle="dotted")
plt.plot(x_traj[:, 1], color="blue", label="True y")
plt.xlabel("Time step")
plt.ylabel("Y component")
plt.legend()
plt.title("Y Component Comparison")
plt.show()

