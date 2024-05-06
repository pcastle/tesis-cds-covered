import matplotlib.pyplot as plt
from sympy import symbols, Piecewise, Min, Max, integrate, pycode, parse_expr, solve, lambdify
# from monopolio import *
import numpy as np


# x = np.linspace(0.25,1,100)
# print(d.subs([(b,0.25),(p,0.7)])/0.7)
# print(u.subs([(b,0.25),(p,0.7)]))

# sympy.plot(u.subs(b,0.62), (p,0.62,1))

y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)
u = integrate(Max(0,Min(1,p*q/b)*y - q)*3*y**2, (y,0,1))

def u_fun(p_v,q_v):
    fun = lambdify([p,q],u.subs(b,0.3))
    return fun(p_v,q_v)

def llega_al_fondo(p,b_v):
    return b_v/p

P = np.linspace(0,1,100)
Q = np.linspace(0,1,100)

P,Q = np.meshgrid(P,Q)

U = np.zeros((100,100))

for x in range(100):
    for y in range(100):
        U[x,y] = u_fun(P[x,y],Q[x,y]) if P[x,y] != 0 else 0


fig, ax = plt.subplots()

ax.contourf(P,Q,U, 10)
ax.plot(np.linspace(0.3,1,100),llega_al_fondo(np.linspace(0.3,1,100),0.3),'k-',label = '$q = \\frac{\\bar b}{p}$')
ax.plot((0.3,0.3),(0,1), 'k--')
ax.set(xlabel = "precio",
       ylabel = "cantidad")
ax.legend()

plt.savefig('figuras/fun_utilidad.eps', format = 'eps')




