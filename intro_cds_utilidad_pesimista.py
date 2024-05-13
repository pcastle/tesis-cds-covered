import json
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import sympy
import matplotlib.pyplot as plt
import numpy as np
import re

with open('equilibrios_intro_cds_2_otra_H3.json', 'r') as f:
    resultado = json.load(f)


b2_v = 0.4

y2, p2, q2, b2, r = sympy.symbols('y2 p2 q2 b2 r', real=True, positive=True)
t2 = sympy.symbols('t2', positive=True)

g2 = 3*(1-t2*3/4)*(y2-1)**2 + 3*3/4*t2*y2**2

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

u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*3*y2**2,(y2,0,1))
u3 = Min(q2,d3)*(r - integrate((1-Min(1,p2/b2*y2))*y2,(y2,0,1)))
# print(u3.subs([(q2,0.5),(b2,0.3),(p2,0.8),(r,0.2)]))

def f2(x, b2_v, r_v):
    fun = u2.subs([(q2,Min(1,d2/p2,b2/p2)),(b2,b2_v),(r,r_v),(p2,x[0])]).doit()
    return -1*fun

def res(p2_v, r_v):
    fun = p2_v + r_v
    return fun

def f3(x, b2_v, p2_v):
    fun1 = (r - integrate((1-Min(1,p2/b2*y2))*y2,(y2,0,1))).subs([(q2,Min(1,d2/p2,b2/p2)),(b2,b2_v),(r,x[0]),(p2,p2_v)]).doit() 
    fun2 =  Min(q2,d3).subs([(q2,Min(1,d2/p2,b2/p2)),(b2,b2_v),(r,x[0]),(p2,p2_v)]).doit() 
    return -1*fun1*fun2


[float(re.search("0.\d+",k).group(0)) for (k,v) in resultado.items() if '.p' in k]
r_lin = [float(re.search("0.\d+",k).group(0)) for (k,v) in resultado.items() if 'r' in k]
p2_opt = [float(v) for (k,v) in resultado.items() if 'r' in k]

p2_lin = [float(re.search("0.\d+",k).group(0)) for (k,v) in resultado.items() if 'p2' in k]
r_opt = [float(v) for (k,v) in resultado.items() if 'p2' in k]


# Un equilibrio se da cuando p2+r <= 1 en ambas mejores respuesta
r_eq = np.zeros(100)
p2_eq = np.zeros(100)

r_max = 0
for idx in range(100):
    if r_lin[idx] + p2_opt[idx] <= 1:
        r_max = r_lin[idx]

print(r_max)
##  
for idx in range(100):
    if idx == 0:
        r_eq[idx] = r_opt[idx] if r_opt[idx] + p2_lin[idx] <= 1 and r_opt[idx] <= r_max else np.nan
        p2_eq[idx] = p2_lin[idx] if r_opt[idx] + p2_lin[idx] <= 1 and r_opt[idx] <= r_max else np.nan
    else: 
        r_eq[idx] = r_opt[idx] if r_opt[idx] + p2_lin[idx] <= 1 and r_opt[idx] <= r_max and r_opt[idx] > r_opt[idx-1] else np.nan
        p2_eq[idx] = p2_lin[idx] if r_opt[idx] + p2_lin[idx] <= 1 and r_opt[idx] <= r_max and r_opt[idx] > r_opt[idx-1] else np.nan



f2_opt = np.zeros(100)
for idx,p2_v in enumerate(p2_opt):
    f2_opt[idx] = -1*f2([p2_v],b2_v,r_lin[idx])


f3_opt = np.zeros(100)
for idx,r_v in enumerate(r_opt):
    f3_opt[idx] = -1*f3([r_v],b2_v,p2_lin[idx])

fig, ax = plt.subplots()
ax.plot(r_lin,f2_opt,'r')
# ax.plot(p2_lin,f3_opt)
ax


plt.show()
