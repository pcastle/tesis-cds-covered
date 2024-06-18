import sympy
from sympy import Derivative, simplify, Piecewise, Min, Max, integrate, pycode, parse_expr, solve
import scipy
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import multiprocessing as mp
from shapely.geometry import LineString
from matplotlib.colors import ListedColormap


y1, p1, q1, b1, y2, p2, q2, b2, r = sympy.symbols('y1 p1 q1 b1 y2 p2 q2 b2 r', real=True, positive=True)
t1, t2 = sympy.symbols('t1 t2', positive=True)

def plotIssuedAmountIntroCDS(b2_v,r_v,g2,n = 100, savePath = 'figuras/intro_cds_q_funcion.eps'):
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

    d2 = Piecewise((1 - r/(p2 +r)*tgorro2_2, (p2 + r <= 1) & (p2*q2 <= b2)),
                (1 - r/(p2 +r)*tgorro2_1, (p2 + r <= 1) & (p2*q2 > b2)),
                (1-tgorro2_2n,(p2 + r > 1) & (p2*q2 <= b2)),
                (1-tgorro2_1n,True))
    d2 = Max(0,Min(1,d2))

    d2_fun = sympy.lambdify([p2, q2], d2.subs([(b2,b2_v),(r,r_v)]) - p2*q2)
    lin_space = np.linspace(0.0001,1,n)
    p2_lin, q2_lin = np.meshgrid(lin_space,lin_space)
    q_function = np.zeros((n,n))


    for ii in range(n):
        for jj in range(n):
            q_function[ii,jj] = d2_fun(p2_lin[ii,jj],q2_lin[ii,jj])


    fig, ax = plt.subplots()
    cmap = ListedColormap(["k", "k"])
    contourPlot = plt.contour(p2_lin, q2_lin, q_function, levels=[0], cmap=cmap)
    points0 = contourPlot.collections[0].get_paths()[0]
    points1 = contourPlot.collections[0].get_paths()[1]
    plt.cla()


    points0 = points0.vertices
    points1 = points1.vertices
    # prices
    x0 = [element[0] for element in points0] # right side
    x1 = [element[0] for element in points1] # left side
    
    # complete de vector of prices
    listPrices = np.linspace(0,1,n)

    newList = []
    for price in listPrices:
        if price < min(x1) or (price > max(x1) and price < min(x0)) or price > max(x0):
            newList.append(price)

    newList = np.concatenate((newList,x1,x0))
    newList = np.sort(newList)

    # allPoints = np.concatenate((points0,points1))

    # Juntar los array del contorno y el linspace
    auxVar = len(x1)
    allPrices = np.concatenate((x1,x0)) 
    newQFunction = np.zeros(len(newList))
    newQFunction2 = np.zeros(len(newList))
    reachFund = np.zeros(len(newList))
    for ii,p_val in enumerate(newList):
        # search the index
        idx = np.where(allPrices == p_val)[0]
        if idx.size > 0:
            if idx < auxVar:
                newQFunction[ii] = points1[idx][0][1]
            else:
                newQFunction[ii] = points0[idx - auxVar][0][1]
        elif p_val < min(x1):
            newQFunction[ii] = 0
        elif p_val < min(x0):
            newQFunction[ii] = 1
        elif p_val > max(x0):
            newQFunction[ii] = 0 

        if p_val < b2_v:
            reachFund[ii] = np.nan
        else:
            reachFund[ii] = b2_v/p_val

        newQFunction2[ii] = min(1,newQFunction[ii],b2_v/p_val)
        
    ax.plot(newList,reachFund,'--', label = '$p_2q_2 = \\bar b_2$')
    ax.plot(newList,newQFunction,'k',label = '$\\min\\left(1,\\frac{d_2}{p_2}\\right)$')
    ax.plot(newList,newQFunction2,':',color = 'tab:red',label = '$\\min\\left(1,\\frac{d_2}{p_2},\\frac{\\bar b_2}{p_2}\\right)$')
    ax.vlines(b2_v,0,1,linestyles='dashed',color= 'green', label= '$p_2 = \\bar b_2$')
    ax.vlines(1-r_v,0,1,linestyle = 'dashed', color = 'orange', label='$p_2 + r = 1$')

    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    order = [1,2,3,0,4]

    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.savefig(savePath)



def plotIssuedAmountCompCDS(b1_v,b2_v,p1_v,q1_v,r_v,g1,g2):
    vf1 = parse_expr((pycode(integrate(Min(p1*y1/b1,1)*g1,(y1,0,1)))).replace("min(1, b1/p1)","(b1/p1)"), locals())
    tgorro1 = solve(vf1 - p1/(p2+r),t1)[0]
    tgorro1_n = solve(vf1 - p1,t1)[0]



    # Emisor 2
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


    # Funciones Alpha (Esto no depende de r)
    alp = solve(vf2_2/p2 - vf1/p1,t2)[0]
    alp_inv = solve(vf2_2/p2 - vf1/p1,t1)[0]

    d2 = Piecewise((1-tgorro2_2n,(tgorro1_n >= 1) & (p2 +r > 1) & (p2*q2 <= b2)),
                (1-tgorro2_1n,(tgorro1_n >= 1) & (p2 +r > 1) & (p2*q2 > b2)),
                (1-r/(p2+r)*tgorro2_2*tgorro1,(tgorro1 >= 1) & (p2 +r <= 1) & (p2*q2 <= b2)),
                (1-r/(p2+r)*tgorro2_1*tgorro1,(tgorro1 >= 1) & (p2 +r <= 1) & (p2*q2 > b2)),
                (integrate(alp_inv,(t2,Min(tgorro2_2,1),1)) + p2/(p2+r)*tgorro2_2*tgorro1,(alp.subs(t1,1) >= 1) & (p2 + r <= 1) & (p2*q2 <= b2)), 
                (integrate(alp_inv,(t2,Min(tgorro2_1,1),1)) + p2/(p2+r)*tgorro2_1*tgorro1,(alp.subs(t1,1) >= 1) & (p2 + r <= 1) & (p2*q2 > b2)), 
                (integrate(alp_inv,(t2,Min(tgorro2_2,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1) + p2/(p2+r)*tgorro2_2*tgorro1, (p2 + r <= 1) & (p2*q2 <= b2)),
                (integrate(alp_inv,(t2,Min(tgorro2_1,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1) + p2/(p2+r)*tgorro2_1*tgorro1, (p2 + r <= 1) & (p2*q2 > b2)),
                (integrate(alp_inv,(t2,Min(tgorro2_2n,1),1)),(alp.subs(t1,1) >= 1) & (p2 +r > 1) & (p2*q2 <= b2)), 
                (integrate(alp_inv,(t2,Min(tgorro2_1n,1),1)),(alp.subs(t1,1) >= 1) & (p2 +r > 1) & (p2*q2 > b2)), 
                (integrate(alp_inv,(t2,Min(tgorro2_2n,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), (p2*q2 <= b2)),
                (integrate(alp_inv,(t2,Min(tgorro2_1n,1),alp.subs(t1,1))) + 1 - alp.subs(t1,1), True))

    d2_fun = sympy.lambdify([p2, q2, b1, b2, q1, p1, r], d2 - p2*q2)
    lin_space = np.linspace(0,1,100)
    p2_lin, q2_lin  = np.meshgrid(lin_space,lin_space)


    res = np.empty((100,100))

    for ii in range(len(lin_space)):
        for jj in range(len(lin_space)):
            res[ii,jj] = d2_fun(p2_lin[ii,jj],q2_lin[ii,jj],b1_v,b2_v,q1_v,p1_v,r_v)

    plt.figure()
    cmap = ListedColormap(["tab:blue", "tab:blue"])
    plt.contour(p2_lin, q2_lin, res, levels=[0], cmap=cmap)
    plt.show()




if __name__ == '__main__':
    # Set values
    b1_v = 0.3
    b2_v = 0.6
    q2_v = 0.4
    p1_v = 0.8
    q1_v = b1_v/p1_v
    r_v = 0.2

    g2 = 3*(1-t2)*(y2-1)**2 + 3*t2*y2**2
    plotIssuedAmountIntroCDS(b2_v,r_v,g2,150)

    g2 = 3*(1-3/4*t2)*(y2-1)**2 + 3*3/4*t2*y2**2
    plotIssuedAmountIntroCDS(b2_v,r_v,g2,150,'figuras/intro_cds_q_funcion2.eps')

    g2 = 9*(1-t2)*(y2-1)**8 + 9*t2*y2**8
    # plotIssuedAmountIntroCDS(0.7,r_v,g2,150,'figuras/intro_cds_q_funcion3.eps')

    # g1 = 3*(1-t1)*(y1-1)**2 + 3*t1*y1**2


# # Esta funcion ya está dividida en r
# d3 = Piecewise((1/(p2+r)*tgorro2_2, (p2 + r <= 1)), (0, True)) # Ya está dividido en r
# d3 = Max(0,Min(1,d3))


# def q_fun(p2_v,b2_v,r_v):
#     return Min(1,b2_v/p2,d2/p2).subs([(b2,b2_v),(r,r_v),(p2,p2_v)])

# def q_fun2(p2_v,b2_v,r_v):
#     return Min(1,d2/p2).subs([(b2,b2_v),(r,r_v),(p2,p2_v)])


# # b2 = 0.4 y r = .2
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.2)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.2)


# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# ax.legend()
# plt.show()

# # b2 = 0.4 y r = .3
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.3)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.3)


# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.legend()
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# plt.show()

# # b2 = 0.4 y r = .4
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.4)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.4)


# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.legend()
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# plt.show()


# # b2 = 0.4 y r = .5
# p2_lin = np.linspace(0.00001,1,100)
# q_val1 = np.zeros(100)
# q_val2 = np.zeros(100)

# for idx, p2_v in enumerate(p2_lin):
#     q_val1[idx] = q_fun(p2_v,0.4,0.5)
#     q_val2[idx] = q_fun2(p2_v,0.4,0.5)

# fig, ax = plt.subplots()
# ax.plot(p2_lin,q_val1,'k', label = 'mín$\\left(1,\\frac{\\bar b_2}{p_2},\\frac{d_2}{p_2}\\right)$')
# ax.plot(p2_lin,q_val2,'k--', label = 'mín$\\left(1,\\frac{d_2}{p_2}\\right)$')
# ax.legend()
# ax.set(xlabel = 'precio $(p_2)$',
#        ylabel = 'cantidad $(q_2)$')
# plt.show()

