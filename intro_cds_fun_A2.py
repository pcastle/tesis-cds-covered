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
b2_v = 0.3

y2, p2, q2, b2, r = sympy.symbols('y2 p2 q2 b2 r', real=True, positive=True)
t2 = sympy.symbols('t2', positive=True)

h2 = 3*y2**2
h3 = 1
# g2 = 3*(1-t2*3/4)*(y2-1)**2 + 3*3/4*t2*y2**2
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


d2 = Piecewise((1 - r/(p2 +r)*tgorro2_2, (p2 + r <= 1)), (1-tgorro2_2n,True))
d2 = Max(0,Min(1,d2))
# Esta funcion ya está dividida en r
d3 = Piecewise((1/(p2+r)*tgorro2_2, (p2 + r <= 1)), (0, True)) # Ya está dividido en r
d3 = Max(0,Min(1,d3))
# print(d2)
# sympy.plot(d2.subs([(r,0.2),(b2,b2_v),(q2,0.4)]),(p2,0,1))

u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*h2,(y2,0,1))
u3 = Min(q2,d3)*(r - integrate((1-Min(1,p2/b2*y2))*h3,(y2,0,1)))
# print(u3.subs([(q2,0.5),(b2,0.3),(p2,0.8),(r,0.2)]))

# Para un mejor performance transformo las funciones sympy en funciones de python
q2_lam = sympy.lambdify([p2,b2,r], Min(1,d2/p2,b2/p2))

d2_fun = sympy.lambdify([p2,q2,b2,r],d2)

u2_fun = sympy.lambdify([p2,q2,b2,r],u2)
u3_fun = sympy.lambdify([r,b2,p2,q2],u3)

def f2(x, b2_v, r_v):
    return -1*u2_fun(x[0],q2_lam(x[0],b2_v,r_v),b2_v,r_v)

def res(p2_v, r_v):
    fun = p2_v + r_v
    return fun

def f3(x,b2_v,p2_v):
    return -1*u3_fun(x[0],b2_v,p2_v,q2_lam(p2_v,b2_v,x[0]))


def busca_valor_inicial(precio_c,fun_objetivo):
    vector_valores = np.linspace(0.001,1,100)
    for x0 in vector_valores:
        if fun_objetivo([x0],b2_v,precio_c) < 1e-4:
            return x0

    return np.nan

def busca_valor_inicial2(precio_c,fun_objetivo):
    vector_valores = np.linspace(0.0001,1,100)
    for x0 in vector_valores:
        if abs(fun_objetivo([x0],b2_v,precio_c)) > 1e-8:
            return x0

    return np.nan

def mejor_p2(x):
    r_v = x[1]
    p2_v = busca_valor_inicial2(r_v,f2) 
    p2_v = x[0] if np.isnan(p2_v) else p2_v


    cons2 = ({"type": "ineq", "fun": lambda x: 1 - res(x[0],r_v)})
    x0_2 = [p2_v]
    result2 = scipy.optimize.minimize(f2,x0_2,args = (b2_v, r_v), bounds=[(b2_v,1)], tol=1e-10, options={"maxiter" : 1000}, method = 'Nelder-Mead')
    output = dict()
    output["r"] = r_v
    output["mejor_respuesta"] = result2.x[0]
    output["flag"] = result2.success
    return output


r_v = 0.356
p2_lin = np.linspace(0.4,1,1000)
f2_vec1 = np.zeros(1000)
f2_vec2 = np.zeros(1000)
# s_vec = np.zeros(100)
# u3_vec = np.zeros(100)

for idx, p2_v in enumerate(p2_lin):
    f2_vec1[idx] = f2([p2_v],b2_v,r_v) if p2_v+r_v <= 1 else np.nan
    f2_vec2[idx] = f2([p2_v],b2_v,r_v) if p2_v+r_v > 1 else np.nan
    # s_vec[idx] = Min(q2,d3).subs([(q2,Min(1,d2/p2,b2/p2)),(b2,b2_v),(r,r_v),(p2,p2_v)]).doit() 
    # u3_vec[idx] = u3.subs([(q2,Min(1,d2/p2,b2/p2)),(b2,b2_v),(r,r_v),(p2,p2_v)]).doit() 

plt.plot(p2_lin,f2_vec1,'k-')
plt.plot(p2_lin,f2_vec2,'k-')
plt.plot(*(0.6532662561856151,f2([0.6532662561856151],b2_v,r_v)),'b.')
# plt.plot(r_lin,s_vec)
# plt.plot(r_lin,u3_vec)

plt.show()


print(mejor_p2([0.6532662561856151,r_v]),busca_valor_inicial2(r_v,f2),mejor_p2([0.40,r_v]))