import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
from shapely.geometry import LineString


# Eequilibrio buscado
b1_lin = [0.3, 0.3, 0.4, 0.3]
b2_lin = [0.3,0.4, 0.4, 0.55]


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

d1 = Piecewise((1-tgorro1,tgorro2 >= 1),(integrate(alp,(t1,Min(tgorro1,1),1)),alp_inv.subs(t2,1) >= 1), (integrate(alp,(t1,Min(tgorro1,1),alp_inv.subs(t2,1))) + 1 - alp_inv.subs(t2,1), True))
d2 = Piecewise((1-tgorro2,tgorro1 >= 1),(integrate(alp_inv,(t2,Min(tgorro2,1),1)),alp.subs(t1,1) >= 1), (integrate(alp_inv,(t2,Min(tgorro2,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), True))

u1 = integrate(Max(0,Min(1,p1*q1/b1)*y1- q1)*g1.subs(t1,1),(y1,0,1))
u2 = integrate(Max(0,Min(1,p2*q2/b2)*y2- q2)*g2.subs(t2,1),(y2,0,1))


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


if __name__ == '__main__':
    for idx in range(len(b1_lin)):

        b1_v = b1_lin[idx]
        b2_v = b2_lin[idx]


        def busca_valor_inicial(b_v,precio_c,fun_objetivo,f_recuadacion) :
            vector_valores = np.linspace(b_v,1,100)
            for x0 in vector_valores:
                if abs(fun_objetivo([x0],b1_v,b2_v,precio_c)) > 1e-4 and f_recuadacion([x0],b1_v,b2_v,precio_c) <= x0:
                    return x0


        def mejor_p1(x):
            p2_v = x[1]

            p1_v = busca_valor_inicial(b1_v,p2_v,f1,res1)

            cons1 = ({"type": "ineq", "fun": lambda x: x[0] - res1(x, b1_v, b2_v, p2_v)})
            
            x0_1 = [p1_v]
        

            # print(f1([0.9],b1_v,b2_v,p2_v))
            result1 = scipy.optimize.minimize(f1,x0_1,args = (b1_v, b2_v, p2_v),constraints=cons1, bounds=[(b1_v,1)], tol=1e-10, options={"maxiter" : 1000})
            output = dict()
            output["p_2"] = p2_v
            output["mejor_respuesta"] = result1.x[0]
            output["flag"] = result1.success
            return output

        def mejor_p2(x):
            p1_v = x[0]
            p2_v = busca_valor_inicial(b2_v,p1_v,f2,res2)

            cons2 = ({"type": "ineq", "fun": lambda x: x[0] - res2(x, b1_v, b2_v, p1_v)})
            x0_2 = [p2_v]
            result2 = scipy.optimize.minimize(f2,x0_2,args = (b1_v, b2_v, p1_v),constraints=cons2, bounds=[(b2_v,1)], tol=1e-10, options={"maxiter" : 1000})
            output = dict()
            output["p_1"] = p1_v
            output["mejor_respuesta"] = result2.x[0]
            output["flag"] = result2.success
            return output



        pool = mp.Pool(processes=12)
        x0_p1 = np.linspace(b1_v,1,100)
        x0_p2 = np.linspace(b2_v,1,100)
        result1 = pool.starmap(mejor_p1, [[(0.9,x0_2)] for x0_2 in x0_p2])
        # [pool.apply(mejor_p1, args = ([0.9,x0_2],)) for x0_2 in x0_p2]

        pool = mp.Pool(processes=12)
        result2 = pool.starmap(mejor_p2, [[(x0_1,0.9)] for x0_1 in x0_p1])
        # [pool.apply(mejor_p2, args = ([x0_1,0.9],)) for x0_1 in x0_p1]

        # Gráfico 
        X1 = [x["p_2"] for x in result1]
        Y1 = [x["mejor_respuesta"] for x in result1]

        # print(result2)
        X2 = [x['mejor_respuesta'] for x in result2]
        Y2 = [x['p_1'] for x in result2]


        fig, ax = plt.subplots()
        ax.plot(X1,Y1,'k' ,label = '$p_1^*(p_2)$')
        ax.plot(X2,Y2, '--',color = "tab:orange" ,label = '$p_2^*(p_1)$')

        first_line = LineString(np.column_stack((X1, Y1)))
        second_line = LineString(np.column_stack((X2, Y2)))
        intersection = first_line.intersection(second_line)
        if intersection.geom_type == 'MultiPoint':
            plt.plot(*LineString(intersection).xy, 'ro')
            print(*LineString(intersection).xy)
        elif intersection.geom_type == 'Point':
            plt.plot(*intersection.xy, 'ro')
            print(*intersection.xy[0],*intersection.xy[1])
        ax.legend()
        ax.set(ylabel = 'precio de $\\mathcal{A}_1 (p_1)$',
                xlabel = 'precio de $\\mathcal{A}_2 (p_2)$')

        # Se añade un grilla
        plt.grid(color = '0.95')
        plt.savefig(f"figuras/mejor_respuesta_b1_{b1_v}_b2_{b2_v}.eps",format = 'eps')

        # Utilidad Esperada en el equilibrio
        # print(-1*f1([*intersection.xy[1]],b1_v,b2_v,*intersection.xy[0]),-1*f2([*intersection.xy[0]],b1_v,b2_v,*intersection.xy[1]))
        print(f"b1 = {b1_v}, b2 = {b2_v} ",
            " u1 = ", -1*f1([*intersection.xy[1]],b1_v,b2_v,*intersection.xy[0]),
            " u2 = ",-1*f2([*intersection.xy[0]],b1_v,b2_v,*intersection.xy[1]),
            ", p1 = ", *intersection.xy[1],",p2 = ",*intersection.xy[0])
        # g benchmark
        # b1 = 0.3, b2 = 0.3   u1 =  0.397181834199632  u2 =  0.357122822247195 , p1 =  0.8406839735728445 p2 =  0.7514937758572061
        # b1 = 0.3, b2 = 0.4   u1 =  0.390360045117187  u2 =  0.136288134187936 , p1 =  0.8241181364857181 p2 =  0.6386992586329342
        # b1 = 0.4, b2 = 0.4   u1 =  0.208269466343068  u2 =  0.112645514904922 , p1 =  0.7046152623467317 p2 =  0.6320681006343835
        # b1 = 0.3, b2 = 0.55   u1 =  0.407383718554955  u2 =  0.00770890824567862 , p1 =  0.8665405125450226 p2 =  0.6834601978318747