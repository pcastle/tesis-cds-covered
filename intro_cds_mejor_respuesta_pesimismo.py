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
    result2 = scipy.optimize.minimize(f2,x0_2,args = (b2_v, r_v), bounds=[(b2_v,1)], tol=1e-10, options={"maxiter" : 1000},method = 'Nelder-Mead')
    output = dict()
    output["r"] = r_v
    output["mejor_respuesta"] = result2.x[0]
    output["flag"] = result2.success
    return output

def mejor_r(x):
    p2_v = x[0]
    r_v = busca_valor_inicial(p2_v,f3)
    r_v = x[1] if np.isnan(r_v) else r_v

    cons3 = ({'type': 'ineq', 'fun': lambda x: 1 - res(p2_v,x[0])})
    x0_3 = [r_v]
    result3 = scipy.optimize.minimize(f3,x0_3,args = (b2_v, p2_v), bounds=[(0,1)], tol=1e-10, options={"maxiter" : 1000},method = 'Nelder-Mead')
    output = dict()
    output["p2"] = p2_v
    output["mejor_respuesta"] = result3.x[0]
    output["flag"] = result3.success
    return output

def correspondencia(x):
    p2_v = x[0]
    r_v = x[1]

    result2 = mejor_p2(x)
    result3 = mejor_r(x)
    return (result2['mejor_respuesta'] - p2_v,result3['mejor_respuesta']- r_v)


def busqueda_equilibrio(x0_1,x0_2,return_dict):
    x0 = [x0_1,x0_2]
    sol = scipy.optimize.fsolve(correspondencia,x0, xtol=1e-10,maxfev=1000000, full_output=True)

    # Guardo el equilibrio solo si converge a una solución
    return_dict[f"equilibrio"] = sol[0] if sol[2] == 1 else np.nan
    print(sol[0],sol[2])
    return return_dict


if __name__ == '__main__':
    pool = mp.Pool(processes=12)
    x0_r = np.linspace(0,0.999,100)
    x0_p2 = np.linspace(b2_v,0.999,100)
    # result2 = [pool.apply(mejor_p2, args = ([1-x0_3,x0_3],)) for x0_3 in x0_r]
    result2 = pool.starmap(mejor_p2, [[(1-x0_3,x0_3)] for x0_3 in x0_r ])

    pool = mp.Pool(processes=12)
    # result3 = [pool.apply(mejor_r, args = ([x0_2,0.1],)) for x0_2 in x0_p2]
    result3 = pool.starmap(mejor_r, [[(x0_2,0.1)] for x0_2 in x0_p2 ])

    # Gráfico 
    X1 = [x["r"] for x in result2]
    Y1 = [x["mejor_respuesta"] for x in result2]

    # print(result2)
    X2 = [x['mejor_respuesta'] for x in result3]
    Y2 = [x['p2'] for x in result3]

    # Un equilibrio se da cuando p2+r <= 1 en ambas mejores respuesta
    r_eq = np.zeros(100)
    p2_eq = np.zeros(100)

    r_max = 0
    p2_max = 0
    for idx in range(100):
        if X1[idx] + Y1[idx] <= 1:
            r_max = X1[idx]
        if Y2[idx] if X2[idx] + Y2[idx] <= 1 and X2[idx] <= r_max and X2[idx] < X2[idx-1]:
            p2_min = Y2[idx]


    print(p2_min)
    ##  
    for idx in range(100):
        if idx == 0:
            r_eq[idx] = X2[idx] if X2[idx] + Y2[idx] <= 1 and X2[idx] <= r_max else np.nan
            p2_eq[idx] = Y2[idx] if X2[idx] + Y2[idx] <= 1 and X2[idx] <= r_max else np.nan
        else:
            r_eq[idx] = X2[idx] if X2[idx] + Y2[idx] <= 1 and X2[idx] <= r_max and X2[idx] < X2[idx-1] else np.nan
            p2_eq[idx] = Y2[idx] if X2[idx] + Y2[idx] <= 1 and X2[idx] <= r_max and X2[idx] < X2[idx-1] else np.nan


    fig, ax = plt.subplots()
    ax.plot(X1,Y1,'k' ,label = '$p_2^*(r)$')
    ax.plot(X2,Y2, 'k--' ,label = '$r^*(p_2)$')
    ax.plot(r_eq,p2_eq,'r',alpha=.9)

    ax.set(ylabel = 'precio de $\\mathcal{A}_2 (p_2)$',
           xlabel = 'precio de $\\mathcal{A}_3 (r)$')

    first_line = LineString(np.column_stack((X1, Y1)))
    second_line = LineString(np.column_stack((X2, Y2)))
    intersection = first_line.intersection(second_line)
    print(intersection)
    # if intersection.geom_type == 'MultiPoint':
    #     plt.plot(*LineString(intersection).xy, 'ro')
    #     print(*LineString(intersection).xy)
    # elif intersection.geom_type == 'Point':
    #     plt.plot(*intersection.xy, 'ro')
    #     print(*intersection.xy[0],*intersection.xy[1])
    ax.legend()

    # Se añade un grilla
    ax.grid(color = '0.95')
    plt.show()


    resultado_1 = {f'r_{x["r"]}' : f'{x["mejor_respuesta"]}' for x in result2}
    resultado_2 = {f'p2_{y["p2"]}' : f'{y["mejor_respuesta"]}' for y in result3}

    resultado = {**resultado_1,**resultado_2}
    # Guardo los datos en json
    with open('equilibrios_intro_cds_2_otra_H3.json', 'w') as f:
        json.dump(resultado, f)



