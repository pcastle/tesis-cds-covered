import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import multiprocessing as mp
from shapely.geometry import LineString
import time



# Eequilibrio buscado
y1, p1, q1, b1, y2, p2, q2, b2, r = sympy.symbols('y1 p1 q1 b1 y2 p2 q2 b2 r', real=True, positive=True)
t1, t2 = sympy.symbols('t1 t2', positive=True)

# Eequilibrio buscado
# b1_vec = [0.3, 0.3, 0.4, 0.3]
# b2_vec = [0.3,0.4, 0.4, 0.55]
b1_vec = [0.3]
b2_vec = [0.3]
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


d1 = Piecewise((1-tgorro1_n,((tgorro2_2 >= 1) & (p2+r > 1))),(1-tgorro1,((tgorro2_2 >= 1) & (p2+r <= 1))),(integrate(alp,(t1,Min(tgorro1,1),1)),alp_inv.subs(t2,1) >= 1), (integrate(alp,(t1,Min(tgorro1,1),alp_inv.subs(t2,1))) + 1 - alp_inv.subs(t2,1), True))
d1 = Max(0,Min(1,d1))

d2 = Piecewise((1-tgorro2_2n,(tgorro1 >= 1) & (p2 +r > 1)),(1-r/(p2+r)*tgorro2_2*tgorro1,(tgorro1 >= 1) & (p2 +r <= 1)),
               (integrate(alp_inv,(t2,Min(tgorro2_2,1),1)) + p2/(p2+r)*tgorro2_2*tgorro1,(alp.subs(t1,1) >= 1) & (p2 + r <= 1)), (integrate(alp_inv,(t2,Min(tgorro2_2,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1) + p2/(p2+r)*tgorro2_2*tgorro1, p2 + r <= 1),
               (integrate(alp_inv,(t2,Min(tgorro2_2n,1),1)),(alp.subs(t1,1) >= 1) & (p2 +r > 1)), (integrate(alp_inv,(t2,Min(tgorro2_2n,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), True))
d2 = Max(0,Min(1,d2))
# Esta funcion ya está dividida en r
d3 = Piecewise((1/(p2+r)*tgorro2_2*tgorro1, (p2 + r <= 1)), (0, True)) # Ya está dividido en r
d3 = Max(0,Min(1/r,d3))

# start = time.time()
# print(d2.subs([(p2,0.4),(p1,0.4),(r,0.1),(b1,b1_vec[0]),(b2,b2_vec[0])]),time.time() - start)
# start = time.time()
# print(d1.subs([(p2,0.4),(p1,0.4),(r,0.1),(b1,b1_vec[0]),(b2,b2_vec[0])]),time.time() - start)
# start = time.time()
# print((r*d3).subs([(p2,0.4),(p1,0.4),(r,0.1),(b1,b1_vec[0]),(b2,b2_vec[0])]),time.time() - start)
# sympy.plot(d2.subs([(r,0.2),(b2,b2_v),(q2,0.4)]),(p2,0,1))

# d2_vec = np.zeros(100)
# for idx, p2_v in enumerate(np.linspace(b2_vec[0],1,100)):
#     d2_vec[idx] = d2.subs([(p2,p2_v),(p1,0.4),(r,0.1),(b1,b1_vec[0]),(b2,b2_vec[0])])

# plt.plot(np.linspace(b2_vec[0],1,100),d2_vec)
# plt.show()

u1 = integrate(Max(0,Min(1,p1*q1/b1)*y1- q1)*3*y1**2,(y1,0,1))
u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*3*y2**2,(y2,0,1))
u3 = Min(q2,d3)*(r - integrate((1-Min(1,p2/b2*y2))*h3,(y2,0,1)))
# print(u3.subs([(q2,0.5),(b2,0.3),(p2,0.8),(r,0.2)]))

q2_lam = sympy.lambdify([p2,b1,b2,p1,r], Min(1,d2/p2,b2/p2))
q1_lam = sympy.lambdify([p2,b1,b2,p1,r], d1/p1)

def f1(x, b1_v, b2_v, p2_v, r_v):
    fun = u1.subs([(q1,q1_lam(p2_v,b1_v,b2_v,x[0],r_v)),(q2,q2_lam(p2_v,b1_v,b2_v,x[0],r_v)),(b1, b1_v),(b2,b2_v),(p2,p2_v),(r,r_v),(p1,x[0])]).doit()
    return -1*fun

def res1(x, b1_v, b2_v, p2_v, r_v):
    fun = d1.subs([(q1,q1_lam(p2_v,b1_v,b2_v,x[0],r_v)),(q2,q2_lam(p2_v,b1_v,b2_v,x[0],r_v)),(b1, b1_v),(b2,b2_v),(p2,p2_v),(r,r_v),(p1,x[0])]).doit()
    return fun

def f2(x, b1_v, b2_v, p1_v, r_v):
    fun = u2.subs([(q2,q2_lam(x[0], b1_v, b2_v, p1_v, r_v)),(q1,q1_lam(x[0], b1_v, b2_v, p1_v, r_v)),(b1, b1_v),(b2,b2_v),(p1,p1_v),(r,r_v),(p2,x[0])]).doit()
    return -1*fun

def res2(x, b1_v, b2_v, p1_v, r_v):
    fun = d2.subs([(q1,q1_lam(x[0], b1_v, b2_v, p1_v, r_v)),(q2,q2_lam(x[0], b1_v, b2_v, p1_v, r_v)),(b1, b1_v),(b2,b2_v),(p1,p1_v),(p2,x[0]),(r,r_v)]).doit()
    return fun

def f3(x, b1_v, b2_v, p1_v ,p2_v):
    # fun1 = (r - integrate((1-Min(1,p2/b2*y2))*h3,(y2,0,1))).subs([(q2,q2_lam(p2_v,b1_v,b2_v,p1_v,x[0])),(b1,b1_v),(b2,b2_v),(r,x[0]),(p2,p2_v),(p1,p1_v)]).doit() 
    # fun2 =  Min(q2,d3).subs([(q2,q2_lam(p2_v,b1_v,b2_v,p1_v,x[0])),(q1,q1_lam(p2_v,b1_v,b2_v,p1_v,x[0])),(b1,b1_v),(b2,b2_v),(r,x[0]),(p2,p2_v),(p1,p1_v)]).doit()
    fun1 = u3.subs([(q2,q2_lam(p2_v,b1_v,b2_v,p1_v,x[0])),(q1,q1_lam(p2_v,b1_v,b2_v,p1_v,x[0])),(b1,b1_v),(b2,b2_v),(r,x[0]),(p2,p2_v),(p1,p1_v)]).doit()
    return -1*fun1

# start = time.time()
# print(f1([0.4],0.3,0.3,0.4,0.1),time.time() - start)
# start = time.time()
# print(f2([0.4],0.3,0.3,0.4,0.1),time.time() - start)
# start = time.time()
# print(f3([0.1],0.3,0.3,0.4,0.4),time.time() - start)


if __name__ == '__main__':
    for idx in range(len(b1_vec)):

        b1_v = b1_vec[idx]
        b2_v = b2_vec[idx]


        def busca_valor_inicial(b_v,precio_c,precio_r,fun_objetivo,f_recuadacion) :
            vector_valores = np.linspace(b_v,1,100)
            for x0 in vector_valores:
                if abs(fun_objetivo([x0],b1_v,b2_v,precio_c,precio_r)) > 1e-4 and f_recuadacion([x0],b1_v,b2_v,precio_c,precio_r) <= x0:
                    return x0
                
        def busca_valor_inicial2(p1_v,p2_v,fun_objetivo):
            vector_valores = np.linspace(0.0001,1-b2_v,100)
            for x0 in vector_valores:
                if fun_objetivo([x0],b1_v,b2_v,p1_v,p2_v) < 1e-4:
                    return x0

            return np.nan
        

        # print(busca_valor_inicial2(0.4,0.4,f3))

        def mejor_p1(x):
            p2_v = x[1]
            r_v = x[2]

            p1_v = busca_valor_inicial(b1_v,p2_v,r_v,f1,res1)

            cons1 = ({"type": "ineq", "fun": lambda x: x[0] - res1(x, b1_v, b2_v, p2_v, r_v)})
            
            x0_1 = [p1_v]
        

            # print(f1([0.9],b1_v,b2_v,p2_v))
            result1 = scipy.optimize.minimize(f1,x0_1,args = (b1_v, b2_v, p2_v,r_v),constraints=cons1, bounds=[(b1_v,1)], tol=1e-10, options={"maxiter" : 1000})
            output = dict()
            output["p_2"] = p2_v
            output["r"] = r_v
            output["mejor_respuesta"] = result1.x[0]
            output["flag"] = result1.success
            return output

        def mejor_p2(x):
            p1_v = x[0]
            r_v = x[2]
            # p2_v = busca_valor_inicial(b2_v,p1_v,r_v,f2,res2)
            p2_v = x[1] 
            

            # La restriccion no es necesaria
            cons2 = ({"type": "ineq", "fun": lambda x: x[0] - res2(x, b1_v, b2_v, p1_v)})
            x0_2 = [p2_v]
            result2 = scipy.optimize.minimize(f2,x0_2,args = (b1_v, b2_v, p1_v, r_v), bounds=[(b2_v,1)], tol=1e-10, options={"maxiter" : 1000},method = 'Nelder-Mead')
            output = dict()
            output["p_1"] = p1_v
            output["r"] = r_v
            output["mejor_respuesta"] = result2.x[0]
            output["flag"] = result2.success
            return output


        def mejor_r(x):
            p1_v = x[0]
            p2_v = x[1]
            r_v = busca_valor_inicial2(p1_v,p2_v,f3)
            r_v = x[2] if np.isnan(r_v) else r_v

            x0_3 = [r_v]
            result3 = scipy.optimize.minimize(f3,x0_3,args = (b1_v, b2_v, p1_v, p2_v), bounds=[(0,1-b2_v)], tol=1e-10, options={"maxiter" : 1000},method = 'Nelder-Mead')
            output = dict()
            output["p_2"] = p2_v
            output["p_1"] = p1_v
            output["mejor_respuesta"] = result3.x[0]
            output["flag"] = result3.success
            return output

        def correspondencia(x):
            p1_v = x[0]
            p2_v = x[1]
            r_v = x[2]

            result1 = mejor_p1(x)
            result2 = mejor_p2(x)
            result3 = mejor_r(x)
            return (result1['mejor_respuesta'] - p1_v,result2['mejor_respuesta'] - p2_v,result3['mejor_respuesta']- r_v)


        def busqueda_equilibrio(x0_1,x0_2,x0_3,return_dict):
            x0 = [x0_1,x0_2,x0_3]
            sol = scipy.optimize.fsolve(correspondencia,x0, xtol=1e-10,maxfev=1000000, full_output=True)

            # Guardo el equilibrio solo si converge a una solución
            return_dict[f"equilibrio"] = sol[0] if sol[2] == 1 else np.nan
            print(sol[0],sol[2])
            return return_dict
        
        # print(mejor_p1([0.4,0.4,0.1]))
        # print(mejor_p2([0.4,0.4,0.1]))
        # print(mejor_r([0.4,0.4,0.1]))
        print(busqueda_equilibrio(0.4,0.4,0.1,{}))
