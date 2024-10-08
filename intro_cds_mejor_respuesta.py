import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import multiprocess as mp
from shapely.geometry import LineString



def plotBestResponceIntroCDS(b2_v,g2,h2,h3,nProcess = 12, plot = True,
                             savePath = 'figuras/intro_cds/', aditional = '',
                             nLinspace = 1000, searchEquilibrium = {}):
    t2 = sympy.symbols('t2', positive=True)
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

    u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*h2,(y2,0,1))
    u3 = Min(q2,d3)*(r - integrate((1-Min(1,p2/b2*y2))*h3,(y2,0,1)))
    # print(u3.subs([(q2,0.5),(b2,0.3),(p2,0.8),(r,0.2)]))

    # Para un mejor performance transformo las funciones sympy en funciones de python
    q2_lam = sympy.lambdify([p2,b2,r], Min(1,d2/p2,b2/p2))

    d2_fun = sympy.lambdify([p2,q2,b2,r],d2)

    u2_fun = sympy.lambdify([p2,q2,b2,r],u2)
    u3_fun = sympy.lambdify([r,b2,p2,q2],u3)

    def f2(x, b2_v, r_v):
        return -1*u2_fun(x[0],q2_lam(x[0],b2_v,r_v),b2_v,r_v)

    def res(p2_v, r_v):
        fun = p2_v + r_v
        return fun

    def f3(x,b2_v,p2_v):
        return -1*u3_fun(x[0],b2_v,p2_v,q2_lam(p2_v,b2_v,x[0]))

    # print(f3([0.01],b2_v,0.1))

    def busca_valor_inicial(precio_c,fun_objetivo):
        vector_valores = np.linspace(0.001,1,100)
        for x0 in vector_valores:
            if fun_objetivo([x0],b2_v,precio_c) < 1e-4:
                return x0

        return np.nan

    def busca_valor_inicial2(precio_c,fun_objetivo):
        vector_valores = np.linspace(0.0001,1,100)
        for x0 in vector_valores:
            if abs(fun_objetivo([x0],b2_v,precio_c)) > 1e-4:
                return x0

        return np.nan

    def mejor_p2(x):
        r_v = x[1]
        p2_v = x[0]

        cons2 = ({"type": "ineq", "fun": lambda x: 1 - res(x[0],r_v)})
        x0_2 = [p2_v]
        # result2 = scipy.optimize.direct(f2,bounds=[(b2_v,1)], args = (b2_v, r_v), maxfun=100000, maxiter = 100000)
        result2 = scipy.optimize.shgo(f2,bounds=[(b2_v,1)], args = (b2_v, r_v), n=64, iters=3)
        # print(result2)
        # result2 = scipy.optimize.minimize(f2,x0_2,args = (b2_v, r_v), bounds=[(0,1)], tol=1e-10, options={"maxiter" : 1000},method = 'Nelder-Mead')
        output = dict()
        output["r"] = r_v
        output["mejor_respuesta"] = result2.x[0]
        output["flag"] = result2.success
        return output

    def mejor_r(x):
        p2_v = x[0]
        r_v = x[1]

        cons3 = ({'type': 'ineq', 'fun': lambda x: 1 - res(p2_v,x[0])})
        x0_3 = [r_v]
        # result3 = scipy.optimize.direct(f3,bounds=[(0,1-b2_v)],args = (b2_v, p2_v), maxfun = 200000,maxiter = 200000)
        result3 = scipy.optimize.shgo(f3,bounds=[(0,1-b2_v)],args = (b2_v, p2_v), n=64, iters=3)
        # result3 = scipy.optimize.minimize(f3,x0_3,args = (b2_v, p2_v), bounds=[(0,1)], tol=1e-10, options={"maxiter" : 1000},method = 'Nelder-Mead')
        # print(result3.x[0],result3.fun)
        # if result3.fun == 0:
        #     print(result3)
        output = dict()
        output["p2"] = p2_v
        output["mejor_respuesta"] = result3.x[0]
        output["flag"] = result3.success
        output["fun"] = result3.fun
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
        return_dict[f"equilibrio"] = sol[0]
        print(sol[0],sol[2])
        return return_dict
    
    if searchEquilibrium:
        equilibrio = busqueda_equilibrio(searchEquilibrium["p2"],searchEquilibrium["r"],{})
        p2_eq = equilibrio["equilibrio"][0]
        r_eq = equilibrio["equilibrio"][1] 
        print(f"p_2 = {p2_eq:0.4f}, r = {r_eq:0.4f}, p_2 +r = {p2_eq + r_eq:0.4f}")

        resumen = {"b2": b2_v,
                   "g2": pycode(g2),
                   "h3": pycode(h3),
                   "p_2": p2_eq,
                   "q_2": q2_lam(p2_eq,b2_v,r_eq),
                   "r": r_eq,
                   "u_2": -1*f2([p2_eq],b2_v,r_eq),
                   "u_3": -1*f3([r_eq],b2_v,p2_eq)}


    
    if plot:
        pool = mp.Pool(processes=nProcess)
        lin_space = np.linspace(0,1,nLinspace)
        x0_r = lin_space[lin_space <= 1-b2_v]
        x0_p2 = lin_space[lin_space >= b2_v]

        result2 = pool.starmap(mejor_p2, [[(0.9,x0_3)] for x0_3 in x0_r ])

        pool = mp.Pool(processes=12)
        result3 = pool.starmap(mejor_r, [[(x0_2,0.1)] for x0_2 in x0_p2 ])

        X1 = [x["r"] if (x['flag']) else np.nan for x in result2]
        Y1 = [x["mejor_respuesta"] if (x['flag'] and x['r']) else np.nan for x in result2]
        Y1_1 = [x["mejor_respuesta"] if (x['flag'] and x['r'] + x['mejor_respuesta'] <= 1) else np.nan for x in result2]
        Y1_2 = [x["mejor_respuesta"] if (x['flag'] and x['r'] + x['mejor_respuesta'] > 1) else np.nan for x in result2]

        # Lo separo en dos para que el gráfico se vea mejor
        X2 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0)) else np.nan for x in result3]
        X2_1 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0) and x['p2'] + x['mejor_respuesta'] <= 1) else np.nan for x in result3]
        X2_2 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0) and x['p2'] + x['mejor_respuesta'] > 1) else np.nan for x in result3]
        Y2 = [x['p2'] if (x['flag'] or x['fun'] == 0)  else np.nan for x in result3]

        fig, ax = plt.subplots()
        ax.plot(X1,Y1_1,'k' ,label = '$p_2^*(r)$')
        ax.plot(X1,Y1_2,'k')
        ax.plot(X2_1,Y2, '--',color = 'tab:orange', label = '$r^*(p_2)$')
        ax.plot(X2_2,Y2, '--',color = 'tab:orange')
        ax.plot(x0_r, 1-x0_r, color = 'tab:green')
        # ax.plot(r_eq,p2_eq,'r',alpha=.9)

        ax.set(ylabel = 'precio de $\\mathcal{A}_2 (p_2)$',
            xlabel = 'precio de $\\mathcal{A}_3 (r)$')

        first_line = LineString(np.column_stack((X1, Y1)))
        second_line = LineString(np.column_stack((X2, Y2)))
        intersection = first_line.intersection(second_line)
        print(intersection)
        if intersection.geom_type == 'MultiPoint':
            x, y = zip(*[(point.x, point.y) for point in intersection.geoms])
            plt.plot(x, y, '.', color = 'tab:red')
            # plt.plot(*LineString(intersection).coords.xy, 'ro')
            print(x, y)
        elif intersection.geom_type == 'Point':
            plt.plot(*intersection.xy, 'ro')
            print(*intersection.xy[0],*intersection.xy[1])
        ax.legend()

        # Se añade un grilla
        ax.grid(color = '0.95')
        plt.savefig(f'figuras/intro_cds/resupuesta_b_{b2_v}{aditional}.eps', format = 'eps')

        resultado_1 = {f'r_{x["r"]}' : f'{x["mejor_respuesta"]}' for x in result2}
        resultado_2 = {f'p2_{y["p2"]}' : f'{y["mejor_respuesta"]}' for y in result3}

        resultado = {**resultado_1,**resultado_2}
        # Guardo los datos en json
        # with open(f'{savePath}equilibrios_intro_cds_b_{b2_v}{aditional}.json', 'w') as f:
        #     json.dump(resultado, f)

    if dict:
        return resumen

if __name__ == '__main__':
    y2, p2, q2, b2, r = sympy.symbols('y2 p2 q2 b2 r', real=True, positive=True)
    t2 = sympy.symbols('t2', positive=True)

    # Equilibrio buscado
    b2_vec = [0.3, 0.4]
    h2 = 3*y2**2
    h3_e1 = 3*y2**2
    h3_e2 = 1

    g2_base = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2
    g2_pesimista = 3*(1-3/4*t2)*(y2-1)**2 + 3*3/4*t2*y2**2

    print(plotBestResponceIntroCDS(b2_vec[0],g2_base,h2,h3_e1,aditional='_base_esc1',plot=False, searchEquilibrium={"p2":0.9,"r":0.1}))
    print(plotBestResponceIntroCDS(b2_vec[1],g2_base,h2,h3_e1,aditional='_base_esc1',plot=False, searchEquilibrium={"p2":0.9,"r":0.1}))
    print(plotBestResponceIntroCDS(b2_vec[0],g2_pesimista,h2,h3_e1,aditional='_pesimista_esc1',plot=False, searchEquilibrium={"p2":0.9,"r":0.1}))
    print(plotBestResponceIntroCDS(b2_vec[1],g2_pesimista,h2,h3_e1,aditional='_pesimista_esc1',plot=False, searchEquilibrium={"p2":0.9,"r":0.1}))

    plotBestResponceIntroCDS(b2_vec[0],g2_base,h2,h3_e2,aditional='_base_esc2',plot=False, searchEquilibrium={"p2":0.9,"r":0.1})
    plotBestResponceIntroCDS(b2_vec[1],g2_base,h2,h3_e2,aditional='_base_esc2',plot=False, searchEquilibrium={"p2":0.9,"r":0.1})
    print(plotBestResponceIntroCDS(b2_vec[0],g2_pesimista,h2,h3_e2,aditional='_pesimista_esc2',plot=False, searchEquilibrium={"p2":0.8,"r":0.2}))
    print(plotBestResponceIntroCDS(b2_vec[1],g2_pesimista,h2,h3_e2,aditional='_pesimista_esc2',plot=False, searchEquilibrium={"p2":0.7,"r":0.3}))



