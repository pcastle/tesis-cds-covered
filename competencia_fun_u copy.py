import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
from shapely.geometry import LineString


y1, p1, q1, b1, y2, p2, q2, b2 = sympy.symbols('y1 p1 q1 b1 y2 p2 q2 b2', real=True, positive=True)
t1, t2 = sympy.symbols('t1 t2', positive=True)

g1 = 3*(1-t1)*(y1-1)**2 + 3*t1*y1**2
g2 = 3*(1-t2*3/4)*(y2-1)**2 + 3*t2*3/4*y2**2

## p*q < b barra y p >= b barra
vf1 = parse_expr((pycode(integrate(Min(p1*y1/b1,1)*g1,(y1,0,1)))).replace("min(1, b1/p1)","(b1/p1)"), locals())

## p*q < b barra y p >= b barra
vf2 = parse_expr((pycode(integrate(Min(p2*y2/b2,1)*g2,(y2,0,1)))).replace("min(1, b2/p2)","(b2/p2)"), locals())

# theta gorro

## p*q < b barra y p >= b barra
tgorro1 = solve(vf1 - p1,t1)[0]


## p*q < b barra y p >= b barra
tgorro2 = solve(vf2 - p2,t2)[0]


alp = solve(vf2/p2 - vf1/p1,t2)[0]
alp_inv = solve(vf2/p2 - vf1/p1,t1)[0]

d1 = Piecewise((integrate(alp,(t1,Min(tgorro1,1),1)),alp_inv.subs(t2,1) >= 1), (integrate(alp,(t1,Min(tgorro1,1),alp_inv.subs(t2,1))) + 1 - alp_inv.subs(t2,1), True))
d2 = Piecewise((integrate(alp_inv,(t2,Min(tgorro2,1),1)),alp.subs(t1,1) >= 1), (integrate(alp_inv,(t2,Min(tgorro2,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), True))

u1 = integrate(Max(0,Min(1,p1*q1/b1)*y1- q1)*3*y1**2,(y1,0,1))
u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*3*y2**2,(y2,0,1))


# Eequilibrios buscados
b1_v1 = 0.3
b2_v1 = 0.3

b1_v2 = 0.3
b2_v2 = 0.4

b1_v3 = 0.4
b2_v3 = 0.4

b1_v4 = 0.3
b2_v4 = 0.55

# Con b1 = b2 = 0.3
p1_opt1 = 0.8407
p2_opt1 = 0.7515

# Con b1 = 0.3 y b2 = 0.4
p1_opt2 = 0.8241
p2_opt2 = 0.6387

# con b1 = b2 = 0.4
p1_opt3 = 0.7046
p2_opt3 = 0.6321

# con b1 = 0.3 y b2 = 0.55s
p1_opt4 = 0.8665
p2_opt4 = 0.6835

def f2(x, b1_v, b2_v, p1_v):
    fun = u2.subs([(q1,d1/p1),(q2,d2/p2),(b1, b1_v),(b2,b2_v),(p1,p1_v),(p2,x[0])])
    return fun

p2s1 = np.linspace(b2_v1,1,100)
p2s2 = np.linspace(b2_v2,1,100)
p2s3 = np.linspace(b2_v3,1,100)
p2s4 = np.linspace(b2_v4,1,100)

f2_vec1 = np.zeros(100)
for idx, p2_v in enumerate(p2s1):
    f2_vec1[idx] = f2([p2_v], b1_v1, b2_v1, p1_opt1)

f2_vec2 = np.zeros(100)
for idx, p2_v in enumerate(p2s2):
    f2_vec2[idx] = f2([p2_v], b1_v2, b2_v2, p1_opt2)

f2_vec3 = np.zeros(100)
for idx, p2_v in enumerate(p2s3):
    f2_vec3[idx] = f2([p2_v], b1_v3, b2_v3, p1_opt3)

f2_vec4 = np.zeros(100)
for idx, p2_v in enumerate(p2s4):
    f2_vec4[idx] = f2([p2_v], b1_v4, b2_v4, p1_opt4)

fig, ax = plt.subplots()
ax.plot(p2s1,f2_vec1,'k' ,label = f'$u_2(p_2,p_1^* = {p1_opt1:.3f})$')
ax.plot(p2s2,f2_vec2,'k--' ,label = f'$u_2(p_2,p_1^* = {p1_opt2:.3f})$')
ax.plot(p2s3,f2_vec3,'k:' ,label = f'$u_2(p_2,p_1^* = {p1_opt3:.3f})$')
ax.plot(p2s4,f2_vec4,'k-.' ,label = f'$u_2(p_2,p_1^* = {p1_opt4:.3f})$')
ax.plot(p2_opt1,f2([p2_opt1],b1_v1,b2_v1,p1_opt1), 'ro')
ax.plot(p2_opt2,f2([p2_opt2],b1_v2,b2_v2,p1_opt2), 'ro')
ax.plot(p2_opt3,f2([p2_opt3],b1_v3,b2_v3,p1_opt3), 'ro')
ax.plot(p2_opt4,f2([p2_opt4],b1_v4,b2_v4,p1_opt4), 'ro')
ax.legend()
ax.set(ylabel = 'utilidad',
       xlabel = 'precio ($p_2$)')

# Se añade un grilla
plt.grid(color = '0.95')
plt.savefig(f"figuras/optimo_escenarios_pesimista.eps",format = 'eps')
