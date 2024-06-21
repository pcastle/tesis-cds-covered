import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp


# setup
y1, p1, q1, b1, y2, p2, q2, b2, r = sympy.symbols('y1 p1 q1 b1 y2 p2 q2 b2 r', real=True, positive=True)
t1, t2 = sympy.symbols('t1 t2', positive=True)

def searchEquilibriumCompCDS(b1_v,b2_v,g1,g2,h1,h2,h3,nProcess = 12,
                             savePath = 'resultados/', aditional = '',
                             nP1 = 5,nP2 = 5,nR = 5):
    print(nP1*nP2*nR,"equilibriums will be calculated")
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
    # No need

    # theta gorro
    ## p*q > b barra 
    tgorro2_1 = solve(vf2_1 - p2/(p2+r),t2)[0]
    tgorro2_1n = solve(vf2_1 - p2,t2)[0]

    ## p*q < b barra y p >= b barra
    tgorro2_2 = solve(vf2_2 - p2/(p2+r),t2)[0]
    tgorro2_2n = solve(vf2_2 - p2,t2)[0]

    ## p*q < b barra y p < b barra
    # No need

    # Funciones Alpha (Esto no depende de r)
    alp = solve(vf2_2/p2 - vf1/p1,t2)[0]
    alp_inv = solve(vf2_2/p2 - vf1/p1,t1)[0]

    d1 = Piecewise((1-tgorro1_n,((tgorro2_2n >= 1) & (p2+r > 1))),(1-tgorro1,((tgorro2_2 >= 1) & (p2+r <= 1))),
                (integrate(alp,(t1,Min(tgorro1,1),1)),(alp_inv.subs(t2,1) >= 1) & (p2 + r <= 1)),
                    (integrate(alp,(t1,Min(tgorro1_n,1),1)),(alp_inv.subs(t2,1) >= 1) & (p2 + r > 1)),
                (integrate(alp,(t1,Min(tgorro1,1),alp_inv.subs(t2,1))) + 1 - alp_inv.subs(t2,1), (p2 + r <= 1)),
                (integrate(alp,(t1,Min(tgorro1_n,1),alp_inv.subs(t2,1))) + 1 - alp_inv.subs(t2,1), True))
    d1 = Max(0,Min(1,d1))

    d2 = Piecewise((1-tgorro2_2n,(tgorro1 >= 1) & (p2 +r > 1)),
                   (1-r/(p2+r)*tgorro2_2*tgorro1,(tgorro1 >= 1) & (p2 +r <= 1)),
                   (integrate(alp_inv,(t2,Min(tgorro2_2,1),1)) + p2/(p2+r)*tgorro2_2*tgorro1,(alp.subs(t1,1) >= 1) & (p2 + r <= 1)),
                   (integrate(alp_inv,(t2,Min(tgorro2_2,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1) + p2/(p2+r)*tgorro2_2*tgorro1, p2 + r <= 1),
                   (integrate(alp_inv,(t2,Min(tgorro2_2n,1),1)),(alp.subs(t1,1) >= 1) & (p2 +r > 1)), 
                   (integrate(alp_inv,(t2,Min(tgorro2_2n,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), True))
    d2 = Max(0,Min(1,d2))
    # Esta funcion ya está dividida en r
    d3 = Piecewise((1/(p2+r)*tgorro2_2*tgorro1, (p2 + r <= 1)), (0, True)) # Ya está dividido en r
    d3 = Max(0,Min(1/r,d3))

    u1 = integrate(Max(0,Min(1,p1*q1/b1)*y1- q1)*3*y1**2,(y1,0,1))
    u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*3*y2**2,(y2,0,1))
    u3 = Min(q2,d3)*(r - integrate((1-Min(1,p2/b2*y2))*h3,(y2,0,1)))
    # print(u3.subs([(q2,0.5),(b2,0.3),(p2,0.8),(r,0.2)]))

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
            if fun_objetivo([x0],b1_v,b2_v,p1_v,p2_v) <= 0:
                return x0

        return np.nan


    def mejor_p1(x):
        p2_v = x[1]
        r_v = x[2]

        p1_v = busca_valor_inicial(b1_v,p2_v,r_v,f1,res1)
        p1_v = x[0]  if np.isnan(p1_v) else p1_v

        cons1 = ({"type": "ineq", "fun": lambda x: x[0] - res1(x, b1_v, b2_v, p2_v, r_v)})
        x0_1 = [p1_v]
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
        p2_v = busca_valor_inicial(b2_v,p1_v,r_v,f2,res2)
        p2_v = x[1]  if np.isnan(p2_v) else p2_v
        
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


    def busqueda_equilibrio(x0_1,x0_2,x0_3):
        x0 = [x0_1,x0_2,x0_3]
        sol = scipy.optimize.fsolve(correspondencia,x0, xtol=1e-10,maxfev=1000000, full_output=True)

        # Guardo el equilibrio solo si converge a una solución
        resultado = sol[0] if sol[2] == 1 else np.nan
        # print(sol[0],sol[2])
        return resultado
    
    pool = mp.Pool(processes=nProcess)
    equilibrio = pool.starmap(busqueda_equilibrio, [[x0_1,x0_2,x0_3] for x0_1 in np.linspace(b1_v,1,nP1) for x0_2 in np.linspace(b2_v,1,nP2) for x0_3 in np.linspace(0,1-b2_v,nR)])
    
    for idx, x in enumerate(equilibrio):
        if not np.all(np.isnan(x)):
            res = busqueda_equilibrio(x[0],x[1],x[2])
            print(res[0],res[1],res[2])
            if np.all(res != x):
                equilibrio[idx] = np.nan
    
    with open(f'{savePath}/equilibrios_competencia_b1_{b1_v}_b2_{b2_v}{aditional}.txt', 'w') as f:
            for s in equilibrio:
                f.write(str(s) + '\n')



if __name__ == '__main__':
    b1_v = 0.3
    b2_v = 0.3
    h1 = 3*y1**2
    h2 = 3*y2**2
    h3_esc1 = 3*y2**2
    h3_esc2 = 1

    g1 = 3*(1-t1)*(y1-1)**2 + 3*t1*y1**2
    g2_base = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2
    g2_pesimista = 3*(1-3/4*t2)*(y2-1)**2 + 3*3/4*t2*y2**2
    
    # searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_base,h1,h2,h3_esc1,nProcess=12,aditional='_base_esc1')
    searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_base,h1,h2,h3_esc2,nProcess=12,aditional='_base_esc2')
    # searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_pesimista,h1,h2,h3_esc1,nProcess=12,aditional='_pesimista_esc1')
    # searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_pesimista,h1,h2,h3_esc2,nProcess=12,aditional='_pesimista_esc2')

    # b2_v = 0.4
    # searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_base,h1,h2,h3_esc1,nProcess=12,aditional='_base_esc1')
    # searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_base,h1,h2,h3_esc2,nProcess=12,aditional='_base_esc2')
    # searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_pesimista,h1,h2,h3_esc1,nProcess=12,aditional='_pesimista_esc1')
    # searchEquilibriumCompCDS(b1_v,b2_v,g1,g2_pesimista,h1,h2,h3_esc2,nProcess=12,aditional='_pesimista_esc2')
    
