import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import multiprocessing as mp
from shapely.geometry import LineString
from matplotlib.colors import ListedColormap



# Eequilibrio buscado
b2_v = 0.4

y2, p2, q2, b2, r = sympy.symbols('y2 p2 q2 b2 r', real=True, positive=True)
t2 = sympy.symbols('t2', positive=True)

g2 = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2

# def plotIssuedAmount():
## p*q > b barra 
vf2_1 = integrate(Min(y2/q2,1)*g2,(y2,0,1))

## p*q < b barra y p >= b barra
vf2_2 = integrate(Min(p2*y2/b2,1)*g2,(y2,0,1))

## p*q < b barra y p < b barra
# No need since vf2_2 already includes this case




# theta gorro
## p*q > b barra 
tgorro2_1 = solve(vf2_1 - p2/(p2+r),t2)[0]
tgorro2_1n = solve(vf2_1 - p2,t2)[0]

## p*q < b barra y p >= b barra
tgorro2_2 = solve(vf2_2 - p2/(p2+r),t2)[0]
tgorro2_2n = solve(vf2_2 - p2,t2)[0]

## p*q < b barra y p < b barra
# No need since tgorro2_2 and tgorro2_2n already includes this case

# p = sympy.plot_implicit(1-tgorro2_1.subs([(r,0.4),(b2,0.3)])-p2*q2,(p2,0,1),(q2,0,1),depth = 3)
# print(p[0].get_points())


d2 = Piecewise((1 - r/(p2 +r)*tgorro2_2, (p2 + r <= 1) & (p2*q2 <= b2)),
               (1 - r/(p2 +r)*tgorro2_1, (p2 + r <= 1) & (p2*q2 > b2)),
               (1-tgorro2_2n,(p2 + r > 1) & (p2*q2 <= b2)),
               (1-tgorro2_1n,True))
# sympy.plot_implicit(d2.subs([(r,0.2),(b2,0.3)])-p2*q2,(p2,0,1),(q2,0,1))
d2 = Max(0,Min(1,d2))

b2_v, r_v = 0.3, 0.4
d2_fun = sympy.lambdify([q2, p2], d2.subs([(b2,b2_v),(r,r_v)]))
lin_space = np.linspace(0,1,100)
p2_lin, q2_lin = np.meshgrid(lin_space,lin_space)
res = np.zeros((100,100))

for ii in range(len(lin_space)):
    for jj in range(len(lin_space)):
        res[ii,jj] = d2_fun(q2_lin[ii,jj],p2_lin[ii,jj])

plt.figure()
cmap = ListedColormap(["tab:blue", "tab:blue"])
plt.contour(q2_lin, p2_lin, res, levels=[0], cmap=cmap)
plt.show()


# Eequilibrio buscado
y1, p1, q1, b1, y2, p2, q2, b2, r = sympy.symbols('y1 p1 q1 b1 y2 p2 q2 b2 r', real=True, positive=True)
t1, t2 = sympy.symbols('t1 t2', positive=True)

# Eequilibrio buscado
# b1_vec = [0.3, 0.3, 0.4, 0.3]
# b2_vec = [0.3,0.4, 0.4, 0.55]
b1_v = 0.3
b2_v = 0.4
h3 = 3*y2**2

g1 = 3*(1-t1)*(y1-1)**2 + 3*t1*y1**2
g2 = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2

# Emisor 1

vf1 = parse_expr((pycode(integrate(Min(p1*y1/b1,1)*g1,(y1,0,1)))).replace("min(1, b1/p1)","(b1/p1)"), locals())
tgorro1 = solve(vf1 - p1/(p2+r),t1)[0]
tgorro1_n = solve(vf1 - p1,t1)[0]



# Emisor 2
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


# Funciones Alpha (Esto no depende de r)
alp = solve(vf2_2/p2 - vf1/p1,t2)[0]
alp_inv = solve(vf2_2/p2 - vf1/p1,t1)[0]

d2 = Piecewise((1-tgorro2_2n,(tgorro1 >= 1) & (p2 +r > 1) & (p2*q2 <= b2)),
               (1-tgorro2_1n,(tgorro1 >= 1) & (p2 +r > 1) & (p2*q2 > b2)),
               (1-r/(p2+r)*tgorro2_2*tgorro1,(tgorro1 >= 1) & (p2 +r <= 1) & (p2*q2 <= b2)),
               (1-r/(p2+r)*tgorro2_1*tgorro1,(tgorro1 >= 1) & (p2 +r <= 1) & (p2*q2 > b2)),
               (integrate(alp_inv,(t2,Min(tgorro2_2,1),1)) + p2/(p2+r)*tgorro2_2*tgorro1,(alp.subs(t1,1) >= 1) & (p2 + r <= 1) & (p2*q2 <= b2)), 
               (integrate(alp_inv,(t2,Min(tgorro2_1,1),1)) + p2/(p2+r)*tgorro2_1*tgorro1,(alp.subs(t1,1) >= 1) & (p2 + r <= 1) & (p2*q2 > b2)), 
               (integrate(alp_inv,(t2,Min(tgorro2_2,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1) + p2/(p2+r)*tgorro2_2*tgorro1, (p2 + r <= 1) & (p2*q2 <= b2)),
               (integrate(alp_inv,(t2,Min(tgorro2_1,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1) + p2/(p2+r)*tgorro2_1*tgorro1, (p2 + r <= 1) & (p2*q2 > b2)),
               (integrate(alp_inv,(t2,Min(tgorro2_2n,1),1)),(alp.subs(t1,1) >= 1) & (p2 +r > 1) & (p2*q2 <= b2)), 
               (integrate(alp_inv,(t2,Min(tgorro2_1n,1),1)),(alp.subs(t1,1) >= 1) & (p2 +r > 1) & (p2*q2 > b2)), 
               (integrate(alp_inv,(t2,Min(tgorro2_2n,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), (p2*q2 <= b2)),
               (integrate(alp_inv,(t2,Min(tgorro2_1n,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), True))

b2_v, r_v, b1_v, q1_v,p1_v = 0.3, 0.1, 0.3, 0.3/0.8,0.8
d2_fun = sympy.lambdify([q2, p2, b1, b2, q1, p1, r], d2)
lin_space = np.linspace(0,1,200)
q2_lin, p2_lin  = np.meshgrid(lin_space,lin_space)









res = np.empty((200,200))

for ii in range(len(lin_space)):
    for jj in range(len(lin_space)):
        res[ii,jj] = d2_fun(q2_lin[ii,jj],p2_lin[ii,jj],b1_v,b2_v,q1_v,p1_v,r_v)

plt.figure()
cmap = ListedColormap(["tab:blue", "tab:blue"])
plt.contour(q2_lin, p2_lin, res, levels=[0], cmap=cmap)
plt.show()

# # Esta funcion ya está dividida en r
# d3 = Piecewise((1/(p2+r)*tgorro2_2, (p2 + r <= 1)), (0, True)) # Ya está dividido en r
# d3 = Max(0,Min(1,d3))


# def q_fun(p2_v,b2_v,r_v):
#     return Min(1,b2_v/p2,d2/p2).subs([(b2,b2_v),(r,r_v),(p2,p2_v)])

# def q_fun2(p2_v,b2_v,r_v):
#     return Min(1,d2/p2).subs([(b2,b2_v),(r,r_v),(p2,p2_v)])


# # b2 = 0.4 y r = .2
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.2)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.2)


# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# ax.legend()
# plt.show()

# # b2 = 0.4 y r = .3
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.3)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.3)


# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.legend()
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# plt.show()

# # b2 = 0.4 y r = .4
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.4)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.4)


# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.legend()
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# plt.show()


# # b2 = 0.4 y r = .5
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.5)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.5)

# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.legend()
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# plt.show()


