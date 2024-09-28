import matplotlib.pyplot as plt
import json
import re
import math
import numpy as np

def loadData(name = 'equilibrios_monopolio.json'):
    """Load data from a json file"""
    with open(name, 'r') as f:
        resultado = json.load(f)

    resultado = {k: resultado[k] for k in sorted(resultado)}

    return resultado

def plotEquilibriumPrices(resultado, path = 'figuras/monopolio/', aditional = ''):
    b = np.zeros(int(len(resultado)/2))
    idx = 0

    b_barra = [float(re.search("0.\d+",k).group(0)) for (k,v) in resultado.items() if '.p' in k]
    p_opt = [v for (k,v) in resultado.items() if '.p' in k]
    q_opt = [v for (k,v) in resultado.items() if '.q' in k]

    # Plot
    fig, ax = plt.subplots()

    # Datos
    ax.plot(b_barra, p_opt,
            color = 'black')

    # Se añade un grilla
    plt.grid(color = '0.95')

    # Se añaden etiquetas
    ax.set(xlabel = "Monto a recaudar: $\\bar b$",
        ylabel = "Precio de equilibrio: $p^*$")

    # configuración de los ejes
    ax.spines[['right','top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.savefig(f'{path}fig_equilibrios_monopolio{aditional}.eps',format='eps')

    fig, ax = plt.subplots()
    q_1 = np.zeros(len(b_barra))
    q_2 = np.zeros(len(b_barra))

    for i in range(len(b_barra)):
        q_1[i] = q_opt[i] if not abs(q_opt[i]*p_opt[i]- b_barra[i]) > 1e-6 else np.nan
        q_2[i] = q_opt[i] if abs(q_opt[i]*p_opt[i]- b_barra[i]) > 1e-6 else np.nan
        if not math.isnan(q_1[i]):
            x=  b_barra[i]
            y = q_1[i]
            idx = i

    q_2[idx] = q_1[idx]
    ax.plot(b_barra, q_1,
            color = 'blue', label = '$p^*q(p^*) = \\bar b$')
    ax.plot(b_barra, q_2,
            color = 'green', label = '$p^*q(p^*) < \\bar b$')
    # ax.plot(x,y,'ro')
    
# Se añade un grilla
    plt.grid(color = '0.95')

    # Se añaden etiquetas
    ax.set(xlabel = "Monto a recaudar: $\\bar b$",
        ylabel = "Cantidad emitida: $q(p^*)$")

    # configuración de los ejes
    ax.spines[['right','top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.legend(loc = 'upper right')
    

    plt.savefig(f'{path}fig_equilibrios_monopolio_cantidad{aditional}.eps',format='eps')


if __name__ == '__main__':
    res1 = loadData('equilibrios_monopolio.json')
    res2 = loadData('equilibrios_monopolio_2.json')

    plotEquilibriumPrices(res1)
    plotEquilibriumPrices(res2,aditional='_2')