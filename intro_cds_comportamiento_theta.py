import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import multiprocessing as mp

# Symbols declaration
y2, p2, q2, b2, r = sympy.symbols('y2 p2 q2 b2 r', real=True, positive=True)
t2 = sympy.symbols('t2', positive=True)

def plotThetaHatTwoIntroCDS(b2_v,q2_v,r_v, g2,savePath = 'figuras/intro_cds_theta_gorro.eps'):
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

    tgorro = Piecewise((tgorro2_1, (p2*q2 > b2) & (p2 + r <= 1)),(tgorro2_2, (p2*q2 <= b2) & (p2+ r <= 1)),
                    (tgorro2_1n, (p2*q2 > b2) & (p2 + r > 1)),(tgorro2_2n, (p2*q2 <= b2) & (p2+ r > 1)))
    tgorro = Min(1,tgorro)

    def tgorro_fun(p2_v,b2_v,q2_v,r_v):
        return tgorro.subs([(b2,b2_v),(q2, q2_v),(r,r_v),(p2,p2_v)])



    p2_lin = np.linspace(0.01,0.9,100,)
    tgorro_vec = np.zeros(100)
    for idx, p2_v in enumerate(p2_lin):
        tgorro_vec[idx] = tgorro_fun(p2_v,b2_v,q2_v=q2_v,r_v=r_v)

    fig, ax = plt.subplots()

    ax.plot(p2_lin,tgorro_vec,'k', label = "$\\hat\\theta_2$")
    ax.spines[['right','top']].set_visible(False)
    ax.vlines(b2_v,0.9*np.min(tgorro_vec),np.max(tgorro_vec),linestyle = 'dashed', label = '$p_2 = \\bar b_2$', color = 'green')
    ax.vlines(b2_v/q2_v,0.9*np.min(tgorro_vec),np.max(tgorro_vec),linestyle = 'dashed', label = '$p_2q_2=\\bar b_2$')
    ax.vlines(1 - r_v,0.9*np.min(tgorro_vec),np.max(tgorro_vec),linestyle = 'dashed', label = '$p_2 + r = 1$', color = 'orange')
    ax.legend()

    plt.savefig(savePath)


if __name__ == '__main__':

    # Set values
    b2_v = 0.3
    q2_v = 0.4
    r_v = 0.2

    g2 = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2
    plotThetaHatTwoIntroCDS(b2_v,q2_v,r_v, g2)

    g2 = 9*(1-t2)*(y2-1)**8 + 9*t2*y2**8
    plotThetaHatTwoIntroCDS(b2_v,q2_v,r_v, g2,'figuras/intro_cds_theta_gorro2.eps')