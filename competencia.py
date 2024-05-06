import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import multiprocessing as mp


# Eequilibrio buscado
b1_v = 0.3
b2_v = 0.4


y1, p1, q1, b1, y2, p2, q2, b2 = sympy.symbols('y1 p1 q1 b1 y2 p2 q2 b2', real=True, positive=True)
t1, t2 = sympy.symbols('t1 t2', positive=True)

g1 = 3*(1-t1)*(y1-1)**2 + 3*t1*y1**2
g2 = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2

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

u1 = integrate(Max(0,Min(1,p1*q1/b1)*y1- q1)*g1.subs(t1,0.5),(y1,0,1))
u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*g2.subs(t2,0.5),(y2,0,1))


def f1(x, b1_v, b2_v, p2_v):
    fun = u1.subs([(q1,d1/p1),(q2,d2/p2),(b1, b1_v),(b2,b2_v),(p2,p2_v),(p1,x[0])])
    return -1*fun

def res1(x, b1_v, b2_v, p2_v):
    fun = d1.subs([(q1,d1/p1),(q2,d2/p2),(b1, b1_v),(b2,b2_v),(p2,p2_v),(p1,x[0])]).doit()
    return fun

def f2(x, b1_v, b2_v, p1_v):
    fun = u2.subs([(q1,d1/p1),(q2,d2/p2),(b1, b1_v),(b2,b2_v),(p1,p1_v),(p2,x[0])])
    return -1*fun

def res2(x, b1_v, b2_v, p1_v):
    fun = d2.subs([(q1,d1/p1),(q2,d2/p2),(b1, b1_v),(b2,b2_v),(p1,p1_v),(p2,x[0])]).doit()
    return fun


def correspondencia(x):
    p1_v = x[0]
    p2_v = x[1]

    cons1 = ({"type": "ineq", "fun": lambda x: x[0] - res1(x, b1_v, b2_v, p2_v)})
    cons2 = ({"type": "ineq", "fun": lambda x: x[0] - res2(x, b1_v, b2_v, p1_v)})
    x0_1 = [p1_v]
    x0_2 = [p2_v]

    # print(f1([0.9],b1_v,b2_v,p2_v))
    result1 = scipy.optimize.minimize(f1,x0_1,args = (b1_v, b2_v, p2_v),constraints=cons1, bounds=[(b1_v,1)], tol=1e-10, options={"maxiter" : 1000})
    result2 = scipy.optimize.minimize(f2,x0_2,args = (b1_v, b2_v, p1_v),constraints=cons2, bounds=[(b2_v,1)], tol=1e-10, options={"maxiter" : 1000})

    return (result1.x[0]- p1_v,result2.x[0] - p2_v)


def busqueda_equilibrio(x0_1,x0_2,return_dict):
    x0 = [x0_1,x0_2]
    sol = scipy.optimize.fsolve(correspondencia,x0, xtol=1e-10,maxfev=1000000, full_output=True)

    # Guardo el equilibrio solo si converge a una solución
    return_dict[f"equilibrio"] = sol[0] if sol[2] == 1 else np.nan
    print(sol[0],sol[2])
    return return_dict


# b1_v, b2_v = 0.4, 0.6
# sol = scipy.optimize.fsolve(correspondencia,[0.8  , 0.8], xtol=1e-10,maxfev=1000000, full_output=True)
# print(sol[2])

pool = mp.Pool(processes=5)
x0_p1 = np.linspace(b1_v,1,5)
x0_p2 = np.linspace(b2_v,1,5)
result = [pool.apply(busqueda_equilibrio, args = (x0_1,x0_2,{})) for x0_1 in x0_p1 for x0_2 in x0_p2]
print(result)

# if __name__== "__main__":
#     # Posible valores iniciales 
#     x0_p1 = np.linspace(b1_v,1,25)
#     x0_p2 = np.linspace(b2_v,1,25)

#     manager = mp.Manager()
#     return_dict = manager.dict()
#     processes = []

#     for x0_1 in x0_p1:
#         for x0_2 in x0_p2:
#             proc = mp.Process(target = busqueda_equilibrio, args = (x0_1,x0_2,return_dict,))
#             proc.start()
#             processes.append(proc)


#     for process in processes:
#         process.join()

#     # diccionario normal
#     resultado = dict()
#     for x,y in return_dict.items():
#         resultado[x] = y

#     with open(f'equilibrios_competencia_{b1_v}_{b2_v}.json', 'w') as f:
#         json.dump(resultado, f)