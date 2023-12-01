import torch
import torch.nn as nn
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from jax import grad
import jax.numpy as jnp
from jax import jit
from optax import adam
import optax

g_const = 9.81
X2, Y2 = 1.0, 0.7

def forward_model(x2: float, v: float, alpha: float):
    """Calculate y2 - final position

    Args:
        x2 (float): _description_
        v (float): _description_
        alpha (float): _description_
    """
    ret = -(g_const/2)*(x2/(v*jnp.cos(alpha))) + x2*jnp.tan(alpha)
    return ret

def loss(params_loos):
    ret = forward_model(X2, params_loos[0], params_loos[1]) - Y2
    ret = ret**2
    return ret


velocity = 0.1
alpha = 0.15
X2_tensor = X2

params = jnp.array([velocity, alpha])
optic = adam(0.001)
opt_state = optic.init(params)
# A simple update loop.
for _ in range(20000):
  grads = grad(loss)(params)
  updates, opt_state = optic.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

print(params)

 