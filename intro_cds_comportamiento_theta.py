import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import multiprocessing as mp


# Eequilibrio buscado
b2_v = 0.3

y2, p2, q2, b2, r = sympy.symbols('y2 p2 q2 b2 r', real=True, positive=True)
t2 = sympy.symbols('t2', positive=True)

g2 = 9*(1-t2)*(y2-1)**8 + 9*t2*y2**8

## p*q > b barra 
vf2_1 = integrate(Min(y2/q2,1)*g2,(y2,0,1))

## p*q < b barra y p >= b barra
vf2_2 = integrate(Min(p2*y2/b2,1)*g2,(y2,0,1))

## p*q < b barra y p < b barra
vf2_3 = integrate(p2*y2/b2*g2,(y2,0,1))



# theta gorro
## p*q > b barra 
tgorro2_1 = solve(vf2_1 - p2/(p2+r),t2)[0]

## p*q < b barra y p >= b barra
tgorro2_2 = solve(vf2_2 - p2/(p2+r),t2)[0]

## p*q < b barra y p < b barra
tgorro2_3 = solve(vf2_3 - p2/(p2+r),t2)[0]

tgorro = Piecewise((tgorro2_1, p2*q2 > b2),(tgorro2_2, (p2*q2 <= b2)))
tgorro = Min(1,tgorro)

def tgorro_fun(p2_v,b2_v,q2_v,r_v):
    return tgorro.subs([(b2,b2_v),(q2, q2_v),(r,r_v),(p2,p2_v)])



p2_lin = np.linspace(0.01,0.9,100,)
tgorro_vec = np.zeros(100)
for idx, p2_v in enumerate(p2_lin):
    tgorro_vec[idx] = tgorro_fun(p2_v,b2_v,q2_v=0.5,r_v=0.1)

fig, ax = plt.subplots()

ax.plot(p2_lin,tgorro_vec,'k', label = "$\\hat\\theta_2$")
ax.spines[['right','top']].set_visible(False)
ax.legend()

plt.show()