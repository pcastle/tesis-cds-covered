import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import multiprocessing as mp
from shapely.geometry import LineString



# Eequilibrio buscado
b2_v = 0.4

y2, p2, q2, b2, r = sympy.symbols('y2 p2 q2 b2 r', real=True, positive=True)
t2 = sympy.symbols('t2', positive=True)

g2 = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2

## p*q > b barra 
vf2_1 = integrate(Min(y2/q2,1)*g2,(y2,0,1))

## p*q < b barra y p >= b barra
vf2_2 = integrate(Min(p2*y2/b2,1)*g2,(y2,0,1))

## p*q < b barra y p < b barra
vf2_3 = integrate(p2*y2/b2*g2,(y2,0,1))



# theta gorro
## p*q > b barra 
tgorro2_1 = solve(vf2_1 - p2/(p2+r),t2)[0]
tgorro2_1n = solve(vf2_1 - p2,t2)[0]

## p*q < b barra y p >= b barra
tgorro2_2 = solve(vf2_2 - p2/(p2+r),t2)[0]
tgorro2_2n = solve(vf2_2 - p2,t2)[0]

## p*q < b barra y p < b barra
tgorro2_3 = solve(vf2_3 - p2/(p2+r),t2)[0]
tgorro2_3n = solve(vf2_3 - p2,t2)[0]


d2 = Piecewise((1 - r/(p2 +r)*tgorro2_3, (p2 + r <= 1) & (p2 < b2)),(1 - r/(p2 +r)*tgorro2_2, (p2 + r <= 1) & (p2 >= b2)),(1-tgorro2_3n, p2 < b2) ,(1-tgorro2_2n,True))
d2 = Max(0,Min(1,d2))
# Esta funcion ya está dividida en r
d3 = Piecewise((1/(p2+r)*tgorro2_2, (p2 + r <= 1)), (0, True)) # Ya está dividido en r
d3 = Max(0,Min(1,d3))


def q_fun(p2_v,b2_v,r_v):
    return Min(1,b2_v/p2,d2/p2).subs([(b2,b2_v),(r,r_v),(p2,p2_v)])

def q_fun2(p2_v,b2_v,r_v):
    return Min(1,d2/p2).subs([(b2,b2_v),(r,r_v),(p2,p2_v)])


# b2 = 0.4 y r = .2
p2_lin = np.linspace(0.00001,1,100)
q_val1 = np.zeros(100)
q_val2 = np.zeros(100)

for idx, p2_v in enumerate(p2_lin):
    q_val1[idx] = q_fun(p2_v,0.4,0.2)
    q_val2[idx] = q_fun2(p2_v,0.4,0.2)


fig, ax = plt.subplots()
ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
ax.set(xlabel = 'precio $(p_2)$',
       ylabel = 'cantidad $(q_2)$')
ax.legend()
plt.show()

# b2 = 0.4 y r = .3
p2_lin = np.linspace(0.00001,1,100)
q_val1 = np.zeros(100)
q_val2 = np.zeros(100)

for idx, p2_v in enumerate(p2_lin):
    q_val1[idx] = q_fun(p2_v,0.4,0.3)
    q_val2[idx] = q_fun2(p2_v,0.4,0.3)


fig, ax = plt.subplots()
ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
ax.legend()
ax.set(xlabel = 'precio $(p_2)$',
       ylabel = 'cantidad $(q_2)$')
plt.show()

# b2 = 0.4 y r = .4
p2_lin = np.linspace(0.00001,1,100)
q_val1 = np.zeros(100)
q_val2 = np.zeros(100)

for idx, p2_v in enumerate(p2_lin):
    q_val1[idx] = q_fun(p2_v,0.4,0.4)
    q_val2[idx] = q_fun2(p2_v,0.4,0.4)


fig, ax = plt.subplots()
ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
ax.legend()
ax.set(xlabel = 'precio $(p_2)$',
       ylabel = 'cantidad $(q_2)$')
plt.show()


# b2 = 0.4 y r = .5
p2_lin = np.linspace(0.00001,1,100)
q_val1 = np.zeros(100)
q_val2 = np.zeros(100)

for idx, p2_v in enumerate(p2_lin):
    q_val1[idx] = q_fun(p2_v,0.4,0.5)
    q_val2[idx] = q_fun2(p2_v,0.4,0.5)

fig, ax = plt.subplots()
ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
ax.legend()
ax.set(xlabel = 'precio $(p_2)$',
       ylabel = 'cantidad $(q_2)$')
plt.show()


