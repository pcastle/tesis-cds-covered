from sympy import symbols, Piecewise, Min, Max, integrate, pycode, parse_expr, solve, lambdify
import scipy
import numpy as np
import multiprocessing as mp
import json
import re

y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)

g = 3*(1-t)*(y-1)**2 + 3*t*y**2

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

u = q*integrate((p/b*y-1)*g.subs(t,0.6),(y,b/p,1))


def f_objetivo(x,b_v):
    fun = lambdify(p, u.subs(b,b_v))
    return -1*fun(x[0])

# print(f_objetivo([0.6],0.5))
# Debido a que estos métodos son sencibles al valor inicial, este debe ser uno tal que la función objetivo no sea 0
def busca_valor_inicial(b_v,fun_objetivo):
    vector_valores = np.linspace(b_v,1,100)
    for x0 in vector_valores:
        if abs(fun_objetivo([x0],b_v)) > 1e-10:
            return x0

def busqueda_equilibrio(b_v,return_dict):
    x0 = busca_valor_inicial(b_v,f_objetivo)
    # x0 = b_v
    resultado = scipy.optimize.minimize(f_objetivo,[x0],args = (b_v),bounds=[(b_v,1)], tol=1e-10, options={"maxiter" : 1000},method='Nelder-Mead')
    return_dict[f"{b_v}.p"] = resultado.x[0]
    return_dict[f"{b_v}.success"] = resultado.success
    return return_dict


# print(busca_valor_inicial(0.745,f_objetivo))
# print(busqueda_equilibrio(0.7,dict()))


if __name__ == '__main__':

    blin_v = np.linspace(.25,.75,100)
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
        print(resultado)

    with open('equilibrios_monopolio.json', 'w') as f:
        json.dump(resultado, f)