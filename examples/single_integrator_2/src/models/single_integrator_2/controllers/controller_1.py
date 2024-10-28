import jax.numpy as jnp
from typing import *
from jax import jit, Array, lax
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


def controller_1() -> ControllerCallable:
    """
    Create a controller for the given dynamics.

    Args:
        #! USER-POPULATE

    Returns:
        controller (Callable): handle to function computing control

    """

    @jit
    def controller(t: float, x: Array) -> ControllerCallableReturns:
        """Computes control input (1x1).

        Args:
            t (float): time in sec
            x (Array): state vector (or estimate if using observer/estimator)

        Returns:
            unom (Array): 1x1 vector
            data: (dict): empty dictionary
        """
        # logging data
        u_nom = [2 * (10.0 - x[0])]
        data = {"u_nom": u_nom}

        return jnp.array(u_nom), data

    return controller
