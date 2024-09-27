import matplotlib.pyplot as plt
from sympy import symbols, Piecewise, Min, Max, integrate, pycode, parse_expr, solve, lambdify
# from monopolio import *
import numpy as np
import sympy

y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)
g = 3*(1 - t)*(y - 1)**2 + 3*t*y**2

def g_fun(y_v,t_v):
    fun = lambdify([y],g.subs(t,t_v))
    return fun(y_v)

def G_fun(y_v,t_v):
    fun = lambdify([y],integrate(g,(y,0,y)).subs(t,t_v))
    return fun(y_v)

fig, ax = plt.subplots()
# plot_g = sympy.plot(g.subs(t,0),(y,0,1),label = '$\\theta = 0$')
Y = np.linspace(0,1,100)
ax.plot(Y, g_fun(Y,0),'k-', label = '$\\theta = 0$')
ax.plot(Y, g_fun(Y,0.5),'k--', label = '$\\theta = 0.5$')
ax.plot(Y, g_fun(Y,1),'k:', label = '$\\theta = 1$')

ax.grid(alpha = 0.9)
ax.legend(loc = 'upper center')
ax.set(xlabel = "$y$",
       ylabel = '$g(y|\\theta)$')
plt.savefig('figuras/monopolio/fun_inversionistas.eps', format = 'eps')

fig, ax = plt.subplots()
# plot_g = sympy.plot(g.subs(t,0),(y,0,1),label = '$\\theta = 0$')
Y = np.linspace(0,1,100)
ax.plot(Y, G_fun(Y,0),'k-', label = '$\\theta = 0$')
ax.plot(Y, G_fun(Y,0.5),'k--', label = '$\\theta = 0.5$')
ax.plot(Y, G_fun(Y,1),'k:', label = '$\\theta = 1$')

ax.grid(alpha = 0.9)
ax.legend(loc = 'upper left')
ax.set(xlabel = "$y$",
       ylabel = '$G(y|\\theta)$')
plt.savefig('figuras/monopolio/fun_inversionistas_acumulado.eps', format = 'eps')

g = 3*(1 - t*3/4)*(y - 1)**2 + 3*t*3/4*y**2

def g_fun(y_v,t_v):
    fun = lambdify([y],g.subs(t,t_v))
    return fun(y_v)

fig, ax = plt.subplots()
# plot_g = sympy.plot(g.subs(t,0),(y,0,1),label = '$\\theta = 0$')
Y = np.linspace(0,1,100)
ax.plot(Y, g_fun(Y,0),'k-', label = '$\\theta = 0$')
ax.plot(Y, g_fun(Y,0.5),'k--', label = '$\\theta = 0.5$')
ax.plot(Y, g_fun(Y,1),'k:', label = '$\\theta = 1$')

ax.grid(alpha = 0.9)
ax.legend(loc = 'upper center')
ax.set(xlabel = "$y$",
       ylabel = '$g(y|\\theta)$')
plt.savefig('figuras/monopolio/fun_inversionistas_2.eps', format = 'eps')

fig, ax = plt.subplots()
# plot_g = sympy.plot(g.subs(t,0),(y,0,1),label = '$\\theta = 0$')
Y = np.linspace(0,1,100)
ax.plot(Y, G_fun(Y,0),'k-', label = '$\\theta = 0$')
ax.plot(Y, G_fun(Y,0.5),'k--', label = '$\\theta = 0.5$')
ax.plot(Y, G_fun(Y,1),'k:', label = '$\\theta = 1$')

ax.grid(alpha = 0.9)
ax.legend(loc = 'upper center')
ax.set(xlabel = "$y$",
       ylabel = '$G(y|\\theta)$')
plt.savefig('figuras/monopolio/fun_inversionistas_acumulado_2.eps', format = 'eps')