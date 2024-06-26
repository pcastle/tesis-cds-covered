from sympy import symbols, Piecewise, Min, Max, integrate, pycode, parse_expr, solve, lambdify
import scipy
import numpy as np
import multiprocessing as mp
import json
import math

y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)

g = 3*(1-t*3/4)*(y-1)**2 + 3*t*3/4*y**2

# Defino solo la parte en la que se encuentra el equilibrio

# Valor esperado por cada bono
# p >= b barra & p*q <= b barra
vf = parse_expr((pycode(integrate(Min(p*y/b,1)*g,(y,0,1)))).replace("min(1, b/p)","(b/p)"), locals())

# Theta barra
tbarra = solve(integrate(y*g,(y,0,1)) - b, t)[0]

# Theta gorro
tgorro = solve(vf - p,t)[0]

# Funcion de recaudacion
d = 1 - Max(tbarra,Min(tgorro,1))

# Disminucion de la dimension del problema
q = Min(1,d/p)

u = integrate((Min(p*q/b,1)*y-q)*3*y**2,(y,b/p,1))


def f_objetivo(x,b_v):
    fun = lambdify(p, u.subs(b,b_v))
    return -1*fun(x[0])

def recaudacion(x,b_v):
    fun = lambdify(p, d.subs(b,b_v))
    return fun(x[0])

# print(f_objetivo([0.6],0.5))
# Debido a que estos métodos son sencibles al valor inicial, este debe ser uno tal que la función objetivo no sea 0
def busca_valor_inicial(b_v,fun_objetivo,f_recuadacion) :
    vector_valores = np.linspace(b_v,1,100)
    for x0 in vector_valores:
        if abs(fun_objetivo([x0],b_v)) > 1e-4 and f_recuadacion([x0],b_v) <= x0:
            return x0


def busqueda_equilibrio(b_v,return_dict):
    x0 = busca_valor_inicial(b_v,f_objetivo,recaudacion)
    # x0 = b_v
    cons = ({'type':'ineq', 'fun' : lambda x: x[0] - recaudacion(x,b_v)})
    resultado = scipy.optimize.minimize(f_objetivo,[x0],args = (b_v),bounds=[(b_v,1)], 
                                        constraints = cons,
                                        tol=1e-10, options={"maxiter" : 1000})
    return_dict[f"{b_v}.p"] = resultado.x[0] if not math.isnan(resultado.x[0]) else np.nan
    return_dict[f"{b_v}.success"] = str(resultado.success)
    # print(f"el equilibrio con {b_v} y {x0} fue: {resultado.x[0]}")
    return return_dict


if __name__ == '__main__':
    blin_v = np.linspace(.25,0.625,100)
    # result_array = np.zeros((100,1))

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for b_v in blin_v:
        proc = mp.Process(target = busqueda_equilibrio, args = (b_v,return_dict,))
        proc.start()
        processes.append(proc)


    for process in processes:
        process.join()

    # diccionario normal
    resultado = dict()
    for x,y in return_dict.items():
        resultado[x] = y

    with open('equilibrios_monopolio_2.json', 'w') as f:
        json.dump(resultado, f)