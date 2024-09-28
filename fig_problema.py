import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from sympy import symbols, Piecewise, Min, Max, integrate, pycode, parse_expr, solve, lambdify
import sympy
from matplotlib.colors import ListedColormap

# Hacer gráfico de la restricción, el equilibrio, isocuanta
def plotCharacterizationEquilibrium(g, h, b_v = 0.4, nLinspace = 200,
                                    path = 'figuras/monopolio/', aditional = ''):

        
    y, p, q, b = symbols('y p q b', real = True, positive = True)
    t = symbols('t', positive = True)

    # Valor esperado por cada bono
    # p*q >= b barra
    vf_1 = integrate(Min(y/q,1)*g,(y,0,1))

    # p*q <= b barra
    vf_2 = integrate(Min(p*y/b,1)*g,(y,0,1))

    # Theta barra
    tbarra = solve(integrate(y*g,(y,0,1)) - b, t)[0]

    # Theta gorro
    tgorro_1 = solve(vf_1 - p,t)[0]
    tgorro_2 = solve(vf_2 - p,t)[0]

    # Funcion de recaudacion
    d = 1 - tgorro_2

    d = Min(1,d)
    # Disminucion de la dimension del problema
    q = Min(1,d/p)

    u = integrate((Min(p*q/b,1)*y-q)*h,(y,b/p,1))

    # Funcion auxiliar
    q_aux = symbols("q_aux", real = True, positive = True)
    u_aux = integrate((p*q_aux/b*y - q_aux)*h, (y,b/p,1))
    
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

    resultado = busqueda_equilibrio(b_v,{})


    fun_u = solve(u_aux.subs(b,b_v) - u_aux.subs([(b,b_v),(p,resultado[f"{b_v}.p"]),
                                                  (q_aux,recaudacion([resultado[f"{b_v}.p"]],b_v)/resultado[f"{b_v}.p"])]),q_aux)[0]



    def llega_al_fondo(p):
        return b_v/p if p >= b_v else np.nan

    llega_al_fondo_vec = np.vectorize(llega_al_fondo)
    isoutilidad1_vec = np.zeros(nLinspace)

    vf_1 = integrate(Min(y/q_aux,1)*g,(y,0,1))

    # p*q <= b barra
    vf_2 = integrate(Min(p*y/b,1)*g,(y,0,1))

    tgorro_1 = solve(vf_1 - p,t)[0]
    tgorro_2 = solve(vf_2 - p,t)[0]

    # Funcion de recaudacion
    d_aux = Piecewise((1 - tgorro_1, p*q_aux > b),
                  (1 - tgorro_2, p*q_aux < b))
    d_aux = Min(1, d_aux)

    d_fun = sympy.lambdify([p, q_aux], d_aux.subs([(b,b_v)]) - p*q_aux)
    lin_space = np.linspace(0.0001,1,nLinspace)
    p_lin, q_lin = np.meshgrid(lin_space,lin_space)
    q_function = np.zeros((nLinspace,nLinspace))

    for ii in range(nLinspace):
        for jj in range(nLinspace):
            q_function[ii,jj] = d_fun(p_lin[ii,jj],q_lin[ii,jj])

    fig, ax = plt.subplots()
    cmap = ListedColormap(["k", "k"])
    contourPlot = plt.contour(p_lin, q_lin, q_function, levels=[0], cmap=cmap)
    points0 = contourPlot.collections[0].get_paths()[0]
    plt.cla()


    points0 = points0.vertices
    # prices
    x0 = [element[0] for element in points0] 
    q_vec = [element[1] for element in points0] 

    # q_vec = np.zeros(nLinspace)

    p_v = np.linspace(b_v,1,nLinspace)

    if resultado[f"{b_v}.p"]*recaudacion([resultado[f"{b_v}.p"]],b_v)/resultado[f"{b_v}.p"] >= b_v:
        for idx, p_vs in enumerate(p_v):
            isoutilidad1_vec[idx] = fun_u.subs(p,p_vs) if p_vs >= resultado[f"{b_v}.p"] else np.nan
    else:
        p_int = 0
        q_int = 0
        x0 = np.concatenate((np.linspace(1,max(x0),50),x0))
        q_vec = np.concatenate((np.zeros(50),q_vec))
        isoutilidad1_vec = np.zeros(len(x0))
        for idx, p_vs in enumerate(x0):
            if p_int == 0 and p_vs*q_vec[idx] > b_v:
                p_int = p_vs
                q_int = q_vec[idx]

            isoutilidad1_vec[idx] = fun_u.subs(p,p_vs) if p_vs*q_vec[idx] <= b_v else np.nan


    fig, ax = plt.subplots(figsize = [16/2.5,4])
    ax.plot(p_v, llega_al_fondo_vec(p_v), 'k',label = '$q =\\frac{\\bar b}{p}$', )
    ax.plot(x0,q_vec, '--', color = 'tab:blue',label = '$q(p)$')
    ax.plot(x0,isoutilidad1_vec, ':', color = 'tab:orange', label = '$u(p,q) = \\bar u$')
    if resultado[f"{b_v}.p"]*recaudacion([resultado[f"{b_v}.p"]],b_v)/resultado[f"{b_v}.p"] >= b_v:
        ax.plot((resultado[f"{b_v}.p"], 1), (recaudacion([resultado[f"{b_v}.p"]],b_v)/resultado[f"{b_v}.p"], recaudacion([resultado[f"{b_v}.p"]],b_v)/resultado[f"{b_v}.p"]), ':', color = 'tab:orange')
    else:
        ax.plot((p_int, 1), (q_int, q_int), ':', color = 'tab:orange')
    ax.scatter(resultado[f"{b_v}.p"],recaudacion([resultado[f"{b_v}.p"]],b_v)/resultado[f"{b_v}.p"],color = 'red')
    plt.xticks(np.arange(0, 1, step=0.20))
    ax.legend()
    # Se añade un grilla
    plt.grid(color = '0.95')

    # Se añaden etiquetas
    ax.set(xlabel = "precio",
        ylabel = "cantidad")

    plt.savefig(f'{path}equilibrio{aditional}.eps',format='eps')


if __name__ == "__main__":
    y, p, q, b = symbols('y p q b', real = True, positive = True)
    t = symbols('t', positive = True)
    
    b_v = 0.4
    gBase = 3*(1-t)*(y-1)**2 + 3*t*y**2
    gPesimista = 3*(1-3/4*t)*(y-1)**2 + 3*3/4*t*y**2

    h = 3*y**2

    # plotCharacterizationEquilibrium(gBase,h,b_v)
    plotCharacterizationEquilibrium(gPesimista,h,b_v,aditional='_2')

