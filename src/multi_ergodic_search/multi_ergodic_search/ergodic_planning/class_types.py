from typing import NamedTuple
import jax.numpy as jnp
class RiccatiInput(NamedTuple):
    Pt: jnp.ndarray
    At: jnp.ndarray
    Bt: jnp.ndarray
    at: jnp.ndarray
    bt: jnp.ndarray
class LinearizedDynamics(NamedTuple):
    At: jnp.ndarray
    Bt: jnp.ndarray
    at: jnp.ndarray
    bt: jnp.ndarray
class ZDynamicsInput(NamedTuple):
    Pt: jnp.ndarray
    rt: jnp.ndarray
    At: jnp.ndarray
    Bt: jnp.ndarray
    bt: jnp.ndarray

class TrajectorySolution(NamedTuple):
    x: jnp.ndarray
    u: jnp.ndarray
    px: list[jnp.ndarray]
class DualVariables(NamedTuple):
    mu: jnp.ndarray
    lam: jnp.ndarray  # if you enable equality constraints later

class BetaCoefficients(NamedTuple):
    x: jnp.ndarray    # global ergodic coefficients, e.g., (M,)
    px: list[jnp.ndarray]  # local / past coefficients, e.g., (K,)  