import matplotlib.pyplot as plt
import json
import re
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

    # Plot
    fig, ax = plt.subplots(figsize = [16/2.5,4])

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

    plt.show()
    # plt.savefig(f'{path}fig_equilibrios_monopolio{aditional}.eps',format='eps')


if __name__ == '__main__':
    res1 = loadData()
    res2 = loadData('equilibrios_monopolio_2.json')

    # plotEquilibriumPrices(res1)
    plotEquilibriumPrices(res2,aditional='_2')