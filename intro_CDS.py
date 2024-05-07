import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import multiprocessing as mp


# Eequilibrio buscado
b2_v = 0.26

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

## p*q < b barra y p >= b barra
tgorro2_2 = solve(vf2_2 - p2/(p2+r),t2)[0]

## p*q < b barra y p < b barra
tgorro2_3 = solve(vf2_3 - p2/(p2+r),t2)[0]

d2 = Piecewise((1 - r/(p2 +r)*tgorro2_3, ((p2 + r <= 1) & (p2 < b2))),(1 - r/(p2 +r)*tgorro2_2, ((p2 + r <= 1) & (p2 >= b2))), (1 - tgorro2_2, True))
d3 = Piecewise((r/(p2+r)*tgorro2_3, ((p2 + r <= 1) & (p2 < b2))),(r/(p2+r)*tgorro2_2, ((p2 + r <= 1) & (p2 >= b2))), (0, True))
print(d2)
sympy.plot(d2.subs([(r,0.2),(b2,b2_v)]),(p2,0,1))

u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*g2.subs(t2,0.5),(y2,0,1))
u3 = Min(q2,d3/r)*integrate()


def f2(x, b2_v, r_v):
    fun = u2.subs([(q2,d2/p2),(b2,b2_v),(r,r_v),(p2,x[0])])
    return -1*fun

def res2(x, b2_v, r_v):
    fun = d2.subs([(q2,d2/p2),(b2,b2_v),(r,r_v),(p2,x[0])]).doit()
    return fun