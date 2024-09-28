from sympy import symbols, Piecewise, Min, Max, integrate, pycode, parse_expr, solve, lambdify
import numpy as np
import multiprocess as mp
import matplotlib.pyplot as plt


def plotCompararisonVarphi(gBase,gComparison, b_v = 0.3, p_v = 0.4, nLinspace = 200,path = 'figuras/monopolio/',
                           aditional = ''):
    t = symbols('t', positive = True)
    y, p, q, b = symbols('y p q b', real = True, positive = True)

    # Defino solo la parte en la que se encuentra el equilibrio

    # Valor esperado por cada bono
    # p >= b barra & p*q <= b barra
    vfBase = parse_expr((pycode(integrate(Min(p*y/b,1)*gBase,(y,0,1)))).replace("min(1, b/p)","(b/p)"), locals())
    vfComparison = parse_expr((pycode(integrate(Min(p*y/b,1)*gComparison,(y,0,1)))).replace("min(1, b/p)","(b/p)"), locals())


    tLinspace = np.linspace(0,1,nLinspace)
    vfArray = np.zeros((nLinspace,2))

    for idx, t_v in enumerate(tLinspace):
        vfArray[idx,0] = vfBase.subs([(t,t_v),(b,b_v),(p,p_v)])
        vfArray[idx,1] = vfComparison.subs([(t,t_v),(b,b_v),(p,p_v)])

    fig, ax = plt.subplots(figsize = [16/2.5,4])

    # Datos
    ax.plot(tLinspace, vfArray[:,0], color = 'tab:blue', label = 'Caso Base')
    ax.plot(tLinspace, vfArray[:,1], color = 'tab:orange', label = 'Caso Pesimista')

    # Se añade un grilla
    plt.grid(color = '0.95')

    # Se añaden etiquetas
    ax.set(xlabel = "Creencia $\\theta$",
        ylabel = f"Valor esperado sobre el bono: $\\varphi(p={p_v:3.2f},\\theta)$")

    # configuración de los ejes
    ax.spines[['right','top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.legend()
    plt.savefig(f'{path}fig_comparacion_varphi{aditional}.eps',format='eps')


if __name__ == "__main__":
    t = symbols('t', positive = True)
    y, p, q, b = symbols('y p q b', real = True, positive = True)

    gBase = 3*(1-t)*(y-1)**2 + 3*t*y**2
    gComparison = 3*(1-3/4*t)*(y-1)**2 + 3*3/4*t*y**2
    plotCompararisonVarphi(gBase,gComparison,b_v = 0.3, p_v = 0.4)

