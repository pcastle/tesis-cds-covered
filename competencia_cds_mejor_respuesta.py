import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import multiprocess as mp
from shapely.geometry import LineString
import time
from matplotlib.colors import ListedColormap

y1, p1, q1, b1, y2, p2, q2, b2, r = sympy.symbols('y1 p1 q1 b1 y2 p2 q2 b2 r', real=True, positive=True)
t1, t2 = sympy.symbols('t1 t2', positive=True)


def plotBestResponceCompCDS(b1_v,b2_v,p1_v, p2_v, r_v,g1,g2,h1,h2,h3, 
                            n_processors = 12,n_lin = 1000, savePath = 'figuras/competencia_cds/',
                            aditional = ''):
    t1, t2 = sympy.symbols('t1 t2', positive=True)
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


    # Funciones Alpha (Esto no depende de r)
    alp = solve(vf2_2/p2 - vf1/p1,t2)[0]
    alp_inv = solve(vf2_2/p2 - vf1/p1,t1)[0]


    d1 = Piecewise((1-tgorro1_n,((tgorro2_2n >= 1) & (p2+r > 1))),(1-tgorro1,((tgorro2_2 >= 1) & (p2+r <= 1))),
                (integrate(alp,(t1,Min(tgorro1,1),1)),(alp_inv.subs(t2,1) >= 1) & (p2 + r <= 1)),
                    (integrate(alp,(t1,Min(tgorro1_n,1),1)),(alp_inv.subs(t2,1) >= 1) & (p2 + r > 1)),
                (integrate(alp,(t1,Min(tgorro1,1),alp_inv.subs(t2,1))) + 1 - alp_inv.subs(t2,1), (p2 + r <= 1)),
                (integrate(alp,(t1,Min(tgorro1_n,1),alp_inv.subs(t2,1))) + 1 - alp_inv.subs(t2,1), True))
    d1 = Max(0,Min(1,d1))   

    d2 = Piecewise((1-tgorro2_2n,(tgorro1 >= 1) & (p2 +r > 1)),(1-r/(p2+r)*tgorro2_2*tgorro1,(tgorro1 >= 1) & (p2 +r <= 1)),
                (integrate(alp_inv,(t2,Min(tgorro2_2,1),1)) + p2/(p2+r)*tgorro2_2*tgorro1,(alp.subs(t1,1) >= 1) & (p2 + r <= 1)), (integrate(alp_inv,(t2,Min(tgorro2_2,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1) + p2/(p2+r)*tgorro2_2*tgorro1, p2 + r <= 1),
                (integrate(alp_inv,(t2,Min(tgorro2_2n,1),1)),(alp.subs(t1,1) >= 1) & (p2 +r > 1)), (integrate(alp_inv,(t2,Min(tgorro2_2n,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), True))
    d2 = Max(0,Min(1,d2))


    # Esta funcion ya está dividida en r
    d3 = Piecewise((1/(p2+r)*tgorro2_2*tgorro1, (p2 + r <= 1)), (0, True)) # Ya está dividido en r
    d3 = Max(0,Min(1/r,d3))

    u1 = integrate(Max(0,Min(1,p1*q1/b1)*y1- q1)*h1,(y1,0,1))
    u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*h2,(y2,0,1))
    u3 = Min(q2,d3)*(r - integrate((1-Min(1,p2/b2*y2))*h3,(y2,0,1)))

    # Para un mejor performance transformo las funciones sympy en funciones de python
    q1_lam = sympy.lambdify([p1,b1,b2,p2,r], d1/p1)
    q2_lam = sympy.lambdify([p2,b1,b2,p1,r], Min(1,d2/p2,b2/p2))

    d1_fun = sympy.lambdify([p1,q1,b1,b2,p2,q2,r],d1)
    d2_fun = sympy.lambdify([p2,q2,b1,b2,p1,q1,r],d2)

    u1_fun = sympy.lambdify([p1,q1,b1,b2,p2,q2,r],u1)
    u2_fun = sympy.lambdify([p2,q2,b1,b2,p1,q1,r],u2)
    u3_fun = sympy.lambdify([r,b1,b2,p1,q1,p2,q2],u3)


    def f1(x, b1_v, b2_v, p2_v, r_v):
        return -1*u1_fun(x[0],q1_lam(x[0],b1_v,b2_v,p2_v,r_v),b1_v,b2_v,p2_v,q2_lam(p2_v,b1_v,b2_v,x[0],r_v),r_v)

    def res1(x,b1_v,b2_v,p2_v,r_v):
        return d1_fun(x[0],q1_lam(x[0],b1_v,b2_v,p2_v,r_v),b1_v,b2_v,p2_v,q2_lam(p2_v,b1_v,b2_v,x[0],r_v),r_v)


    def f2(x, b1_v, b2_v, p1_v, r_v):
        return -1*u2_fun(x[0],q2_lam(x[0],b1_v,b2_v,p1_v,r_v),b1_v,b2_v,p1_v,q1_lam(p1_v,b1_v,b2_v,x[0],r_v),r_v)

    def res2(x, b1_v, b2_v, p1_v, r_v):
        return d2_fun(x[0],q2_lam(x[0],b1_v,b2_v,p1_v,r_v),b1_v,b2_v,p1_v,q1_lam(p1_v,b1_v,b2_v,x[0],r_v),r_v)

    def f3(x,b1_v,b2_v,p1_v,p2_v):
        return -1*u3_fun(x[0],b1_v,b2_v,p1_v,q1_lam(p1_v,b1_v,b2_v,p2_v,x[0]),p2_v,q2_lam(p2_v,b1_v,b2_v,p1_v,x[0]))

    def busca_valor_inicial(b_v,precio_c,precio_r,fun_objetivo,f_recuadacion) :
        vector_valores = np.linspace(b_v,1,100)
        for x0 in vector_valores:
            if abs(fun_objetivo([x0],b1_v,b2_v,precio_c,precio_r)) > 1e-4 and f_recuadacion([x0],b1_v,b2_v,precio_c,precio_r) <= x0:
                return x0
            
        return np.nan
            
    def busca_valor_inicial2(p1_v,p2_v,fun_objetivo):
        vector_valores = np.linspace(0.0001,1-b2_v,100)
        for x0 in vector_valores:
            if fun_objetivo([x0],b1_v,b2_v,p1_v,p2_v) < 1e-4:
                return x0

        return np.nan

    def mejor_p1(x):
        p2_v = x[1]
        r_v = x[2]

        # p1_v = busca_valor_inicial(b1_v,p2_v,r_v,f1,res1)
        p1_v = x[0]  #if np.isnan(p1_v) else p1_v

        cons1 = ({"type": "ineq", "fun": lambda x: x[0] - res1(x, b1_v, b2_v, p2_v, r_v)})
        x0_1 = [p1_v]
        # result1 = scipy.optimize.minimize(f1,x0_1,args = (b1_v, b2_v, p2_v,r_v),constraints=cons1, bounds=[(b1_v,1)], tol=1e-10, options={"maxiter" : 1000})
        result1 = scipy.optimize.shgo(f1,bounds=[(b1_v,1)],args = (b1_v, b2_v, p2_v,r_v),constraints=cons1, n=64, iters=3)
        output = dict()
        output["p_2"] = p2_v
        output["r"] = r_v
        output["mejor_respuesta"] = result1.x[0]
        output["flag"] = result1.success
        output["fun"] = result1.fun
        return output

    def mejor_p2(x):
        p1_v = x[0]
        r_v = x[2]
        # p2_v = busca_valor_inicial(b2_v,p1_v,r_v,f2,res2)
        p2_v = x[1] # if np.isnan(p2_v) else p2_v
        
        # La restriccion no es necesaria
        cons2 = ({"type": "ineq", "fun": lambda x: x[0] - res2(x, b1_v, b2_v, p1_v)})
        x0_2 = [p2_v]
        # result2 = scipy.optimize.minimize(f2,x0_2,args = (b1_v, b2_v, p1_v, r_v), bounds=[(b2_v,1)], tol=1e-10, options={"maxiter" : 1000},method = 'Nelder-Mead')
        result2 = scipy.optimize.shgo(f2,bounds=[(b2_v,1)], args = (b1_v, b2_v, p1_v, r_v), n=64, iters=3)
        output = dict()
        output["p_1"] = p1_v
        output["r"] = r_v
        output["mejor_respuesta"] = result2.x[0]
        output["flag"] = result2.success
        output["fun"] = result2.fun
        return output


    def mejor_r(x):
        p1_v = x[0]
        p2_v = x[1]
        # r_v = busca_valor_inicial2(p1_v,p2_v,f3)
        # r_v = x[2] if np.isnan(r_v) else r_v
        r_v = x[2]
        x0_3 = [r_v]
        # result3 = scipy.optimize.minimize(f3,x0_3,args = (b1_v, b2_v, p1_v, p2_v), bounds=[(0,1-b2_v)], tol=1e-10, options={"maxiter" : 1000},method = 'Powell')
        result3 = scipy.optimize.shgo(f3,bounds=[(0,1-b2_v)],args = (b1_v, b2_v, p1_v, p2_v), n=64, iters=3)
        output = dict()
        output["p_2"] = p2_v
        output["p_1"] = p1_v
        output["mejor_respuesta"] = result3.x[0]
        output["flag"] = result3.success
        output["fun"] = result3.fun
        return output

    def correspondencia(x):
        p1_v = x[0]
        p2_v = x[1]
        r_v = x[2]

        result1 = mejor_p1(x)
        result2 = mejor_p2(x)
        result3 = mejor_r(x)
        return (result1['mejor_respuesta'] - p1_v,result2['mejor_respuesta'] - p2_v,result3['mejor_respuesta']- r_v)


    def busqueda_equilibrio(x0_1,x0_2,x0_3):
        x0 = [x0_1,x0_2,x0_3]
        sol = scipy.optimize.fsolve(correspondencia,x0, xtol=1e-10,maxfev=1000000, full_output=True)

        # Guardo el equilibrio solo si converge a una solución
        resultado = sol[0]
        return resultado
    
    # Find better responses
    lin_space = np.linspace(0,1,n_lin)
    x0_r = lin_space[lin_space <= 1-b2_v]
    x0_p2 = lin_space[lin_space >= b2_v]
    x0_p1 = lin_space[lin_space >= b1_v]
    
    print("Initial values (p1,p2,r) =(",p1_v,p2_v,r_v,
          ") equilibrium", busqueda_equilibrio(p1_v,p2_v,r_v))

    print("Calculando A2 Fijando p_1 =",p1_v)
    pool = mp.Pool(processes=n_processors)
    # result2 = [pool.apply(mejor_p2, args = ([1-x0_3,x0_3],)) for x0_3 in x0_r]
    result2 = pool.starmap(mejor_p2, [[(p1_v,0.9,x0_3)] for x0_3 in x0_r ])

    print("Calculando A3 Fijando p_1 =",p1_v)
    pool = mp.Pool(processes=n_processors)
    # result3 = [pool.apply(mejor_r, args = ([x0_2,0.1],)) for x0_2 in x0_p2]
    result3 = pool.starmap(mejor_r, [[(p1_v,x0_2,0.1)] for x0_2 in x0_p2 ])

    X1 = [x["r"] if (x['flag'] and x['r']) else np.nan for x in result2]
    Y1 = [x["mejor_respuesta"] if (x['flag'] and x['r']) else np.nan for x in result2]
    Y1_1 = [x["mejor_respuesta"] if (x['flag'] and x['r'] + x['mejor_respuesta'] <= 1) else np.nan for x in result2]
    Y1_2 = [x["mejor_respuesta"] if (x['flag'] and x['r'] + x['mejor_respuesta'] > 1) else np.nan for x in result2]

    # print(result2)
    X2 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0)) else np.nan for x in result3]
    X2_1 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0) and x['p_2'] + x['mejor_respuesta'] <= 1) else np.nan for x in result3]
    X2_2 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0) and x['p_2'] + x['mejor_respuesta'] > 1) else np.nan for x in result3]
    Y2 = [x['p_2'] if (x['flag'] or x['fun'] == 0)  else np.nan for x in result3]

    fig, ax = plt.subplots()
    ax.plot(X1,Y1_1,'k' ,label = f'$p_2^*(p_1={p1_v:5.4f},r)$')
    ax.plot(X1,Y1_2,'k')
    ax.plot(X2_1,Y2, '--',color = 'tab:green', label = f'$r^*(p_1={p1_v:5.4f},p_2)$')
    ax.plot(X2_2,Y2, '--',color = 'tab:green')
    # ax.plot(x0_r, 1-x0_r, color = 'tab:green')
    # ax.plot(r_eq,p2_eq,'r',alpha=.9)

    ax.set(ylabel = 'precio de $\\mathcal{A}_2 (p_2)$',
        xlabel = 'precio de $\\mathcal{A}_3 (r)$')

    first_line = LineString(np.column_stack((X1, Y1)))
    second_line = LineString(np.column_stack((X2, Y2)))
    intersection = first_line.intersection(second_line)
    try:
        print(intersection)
        if intersection.geom_type == 'MultiPoint':
            x, y = zip(*[(point.x, point.y) for point in intersection.geoms])
            plt.plot(x, y, '.', color = 'tab:red')
            # plt.plot(*LineString(intersection).coords.xy, 'ro')
            print(x, y)
        elif intersection.geom_type == 'Point':
            plt.plot(*intersection.xy, 'ro')
            print(*intersection.xy[0],*intersection.xy[1])
    except:
        print("No hay intersección")
    ax.legend()
    plt.plot(*(r_v,p2_v),'b.')
    ax.grid(color = '0.95')
    plt.savefig(f'{savePath}mejor_respuesta_b1_{b1_v}_b2_{b2_v}_1{aditional}.eps',format='eps')

    print("Calculando A1 Fijando p_2 =", p2_v)
    pool = mp.Pool(processes=12)
    result1 = pool.starmap(mejor_p1, [[(0.9,p2_v,x0_3)] for x0_3 in x0_r ])

    print("Calculando A3 Fijando p_2 =", p2_v)
    pool = mp.Pool(processes=12)
    result3 = pool.starmap(mejor_r, [[(x0_1,p2_v,0.1)] for x0_1 in x0_p1 ])

    X1 = [x["r"] if (x['flag']) else np.nan for x in result1]
    Y1 = [x["mejor_respuesta"] if (x['flag']) else np.nan for x in result1]

    # print(result2)
    X2 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0)) else np.nan for x in result3]
    X2_1 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0) and x['p_2'] + x['mejor_respuesta'] <= 1) else np.nan for x in result3]
    X2_2 = [x['mejor_respuesta'] if ((x['flag'] or x['fun'] == 0) and x['p_2'] + x['mejor_respuesta'] > 1) else np.nan for x in result3]
    Y2 = [x['p_1'] if (x['flag'] or x['fun'] == 0)  else np.nan for x in result3]

    fig, ax = plt.subplots()

    ax.plot(X1,Y1,'k' ,label = f'$p_1^*(p_2={p2_v:5.4f},r)$')
    ax.plot(X2_1,Y2, '--',color = 'tab:green', label = f'$r^*(p_1,p_2={p2_v:5.4f})$')
    ax.plot(X2_2,Y2, '--',color = 'tab:green')
    # ax.plot(x0_r, 1-x0_r, color = 'tab:green')

    ax.set(ylabel = 'precio de $\\mathcal{A}_1 (p_1)$',
        xlabel = 'precio de $\\mathcal{A}_3 (r)$')

    first_line = LineString(np.column_stack((X1, Y1)))
    second_line = LineString(np.column_stack((X2, Y2)))
    try:
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
    except:
        print("No hay intersección")
    ax.legend()
    plt.plot(*(r_v,p1_v),'b.')

    # Se añade un grilla
    ax.grid(color = '0.95')
    # plt.show()
    plt.savefig(f'{savePath}mejor_respuesta_b1_{b1_v}_b2_{b2_v}_2{aditional}.eps',format='eps')

    print("Calculando A1 Fijando r =", r_v)
    pool = mp.Pool(processes=12)
    result1 = pool.starmap(mejor_p1, [[(0.9,x0_2,r_v)] for x0_2 in x0_p2 ])

    print("Calculando A2 Fijando r =", r_v)
    pool = mp.Pool(processes=12)
    result2 = pool.starmap(mejor_p2, [[(x0_1,0.9,r_v)] for x0_1 in x0_p1 ])

    X1 = [x["p_2"] if (x['flag']) else np.nan for x in result1]
    Y1 = [x["mejor_respuesta"] if (x['flag']) else np.nan for x in result1]

    # print(result2)
    Y2 = [x["p_1"] if (x['flag']) else np.nan for x in result2]
    X2 = [x["mejor_respuesta"] if (x['flag'] and x['p_1']) else np.nan for x in result2]
    X2_1 = [x["mejor_respuesta"] if (x['flag'] and x['p_1'] + x['mejor_respuesta'] <= 1) else np.nan for x in result2]
    X2_2 = [x["mejor_respuesta"] if (x['flag'] and x['p_1'] + x['mejor_respuesta'] > 1) else np.nan for x in result2]

    fig, ax = plt.subplots()

    first_line = LineString(np.column_stack((X1, Y1)))
    second_line = LineString(np.column_stack((X2, Y2)))
    intersection = first_line.intersection(second_line)
    try:
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
    except:
        print("No hay intersección")
    # separate discrete jumps
    Y1_aux = np.zeros(len(Y1))
    X2_aux = np.zeros(len(X2))
    aux_value = 1
    for idx in range(len(Y1)):
        if Y1[idx] <= aux_value:
            Y1_aux[idx] = Y1[idx]
            aux_value = Y1[idx]
            Y1[idx] = np.nan
        else:
            Y1_aux[idx] = np.nan
    for idx in range(len(X2)):
        if X2[idx] > 1- r_v:
            X2_aux[idx] = X2[idx]
            X2[idx] = np.nan 
        else:
            X2_aux[idx] = np.nan


    ax.plot(X1,Y1,'k' ,label = f'$p_1^*(p_2,r = {r_v:5.4f})$')
    ax.plot(X1,Y1_aux,'k')
    ax.plot(X2,Y2, '--',color = 'orange' ,label = f'$p_2^*(p_1,r = {r_v:5.4f})$')
    ax.plot(X2_aux,Y2, '--',color = 'orange')
    # ax.plot(r_eq,p2_eq,'r',alpha=.9)

    ax.set(ylabel = 'precio de $\\mathcal{A}_1 (p_1)$',
        xlabel = 'precio de $\\mathcal{A}_2 (p_2)$')

    ax.legend()
    plt.plot(*(p2_v,p1_v),'b.')

    # Se añade un grilla
    ax.grid(color = '0.95')
    # plt.show()
    plt.savefig(f'{savePath}mejor_respuesta_b1_{b1_v}_b2_{b2_v}_3{aditional}.eps',format='eps')

    result_dict = {"b1": b1_v,
                   "b2": b2_v,
                   "p1": p1_v,
                   "p2": p2_v,
                   "r": r_v,
                   "u1": -1*f1([p1_v],b1_v,b2_v,p2_v,r_v),
                   "u2": -1*f2([p2_v],b1_v,b2_v,p1_v,r_v),
                   "u3": -1*f3([r_v],b1_v,b2_v,p1_v,p2_v)}
    return result_dict



if __name__ == '__main__':
    h1 = 3*y1**2
    h2 = 3*y2**2
    h3_esc1 = 3*y2**2
    h3_esc2 = 1

    g1 = 3*(1-t1)*(y1-1)**2 + 3*t1*y1**2
    g2_base = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2
    g2_pesimista = 3*(1-3/4*t2)*(y2-1)**2 + 3*3/4*t2*y2**2

    b1_v = 0.3
    b2_v = 0.3
    result = dict()
    
    result["res1"] = plotBestResponceCompCDS(b1_v,b2_v,0.86618168, 0.93785456, 0.06214544,g1,g2_base,h1,h2,h3_esc1,n_lin=1200,
                            aditional='_base_esc1')
    result["res2"] = plotBestResponceCompCDS(b1_v,b2_v,0.8383179182521371, 0.8383179188267246, 0.7,g1,g2_base,h1,h2,h3_esc2,n_lin=1200,
                            aditional='_base_esc2')
    result["res3"] = plotBestResponceCompCDS(b1_v,b2_v,0.87028662, 0.95057993, 0.04942007,g1,g2_pesimista,h1,h2,h3_esc1,n_lin=1200,
                            aditional='_pesimista_esc1')
    result["res4"] = plotBestResponceCompCDS(b1_v,b2_v,0.84078492, 0.75172118, 0.24827882,g1,g2_pesimista,h1,h2,h3_esc2,n_lin=1200,
                            aditional='_pesimista_esc2')
 
    b1_v = 0.3
    b2_v = 0.4
    result["res5"] = plotBestResponceCompCDS(b1_v,b2_v,0.86901544, 0.9481317,  0.0518683 ,g1,g2_base,h1,h2,h3_esc1,n_lin=1200,
                            aditional='_base_esc1')
    result["res6"] = plotBestResponceCompCDS(b1_v,b2_v,0.8030591,  0.70824302, 0.29175698,g1,g2_base,h1,h2,h3_esc2,n_lin=1200,
                            aditional='_base_esc2')
    result["res7"] = plotBestResponceCompCDS(b1_v,b2_v,0.86032267, 0.75856898, 0.24143102,g1,g2_pesimista,h1,h2,h3_esc1,n_lin=1200,
                            aditional='_pesimista_esc1')
    result["res8"] = plotBestResponceCompCDS(b1_v,b2_v,0.82285822, 0.63512424, 0.36487576,g1,g2_pesimista,h1,h2,h3_esc2,n_lin=1200,
                            aditional='_pesimista_esc2')

    print(result)
# {   'res1': 
#  {'b1': 0.3, 'b2': 0.3, 'p1': 0.86618168, 'p2': 0.93785456, 'r': 0.06214544, 
# 'u1': 0.4072497275936395, 'u2': 0.432738455453565, 'u3': 0.01726154454643499},
#     'res2': 
#  {'b1': 0.3, 'b2': 0.3, 'p1': 0.8383179182521371, 'p2': 0.8383179188267246, 'r': 0.1627030303030303, 
# 'u1': 0.3962405932271251, 'u2': 0.3962405955910888, 'u3': -0.0},
#     'res3': 
#  {'b1': 0.3, 'b2': 0.3, 'p1': 0.87028662, 'p2': 0.95057993, 'r': 0.04942007,
#  'u1': 0.40881598503055816, 'u2': 0.4368832904913984, 'u3': 0.013116709508601608}, 
#     'res4': 
#  {'b1': 0.3, 'b2': 0.3, 'p1': 0.84078492, 'p2': 0.75172118, 'r': 0.24827882,
#  'u1': 0.39724269695621217, 'u2': 0.357257446762147, 'u3': 0.019450064088554667},
#     'res5': 
#  {'b1': 0.3, 'b2': 0.4, 'p1': 0.86901544, 'p2': 0.9481317, 'r': 0.0518683,
#  'u1': 0.4083324357608367, 'u2': 0.3360373188708761, 'u3': 0.013962681129123845}, 
#     'res6': 
#  {'b1': 0.3, 'b2': 0.4, 'p1': 0.8030591, 'p2': 0.70824302, 'r': 0.29175698,
#  'u1': 0.38129743441804304, 'u2': 0.21065822173833848, 'u3': 0.002810412369153256},
#     'res7': 
#  {'b1': 0.3, 'b2': 0.4, 'p1': 0.86032267, 'p2': 0.75856898, 'r': 0.24143102,
#  'u1': 0.40499002007860024, 'u2': 0.2420198969317361, 'u3': 0.10798010306826382}, 
#     'res8': 
#  {'b1': 0.3, 'b2': 0.4, 'p1': 0.82285822, 'p2': 0.63512424, 'r': 0.36487576,
#  'u1': 0.3898341190790653, 'u2': 0.15953388906752364, 'u3': 0.01795900972390082}}