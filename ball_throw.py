import torch
import torch.nn as nn
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


g_const = 9.81
X2, Y2 = 1.0, 0.7

def forward_model(x2: float, v: float, alpha: float):
    """Calculate y2 - final position

    Args:
        x2 (float): _description_
        v (float): _description_
        alpha (float): _description_
    """
    ret = -(g_const/2)*(x2/(v*torch.cos(alpha))) + x2*torch.tan(alpha)
    return ret

def loss(v: float, alpha: float):
    ret = forward_model(X2, v, alpha) - Y2
    ret = ret**2
    return ret

 

velocity = torch.tensor(0.1, requires_grad=True)
alpha = torch.tensor(0.1, requires_grad=True)
X2_tensor = torch.tensor(X2)

optic = torch.optim.Adam([velocity, alpha])
loss_hyst = np.array((20000))
for i in range(20000):
    optic.zero_grad()
    forward_model_calc = loss(velocity, alpha)
    forward_model_calc.backward()
    optic.step()

print([velocity, alpha])

 