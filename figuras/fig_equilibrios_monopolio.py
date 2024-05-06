import matplotlib.pyplot as plt
import json
import re
import numpy as np

with open('equilibrios_monopolio_2.json', 'r') as f:
    resultado = json.load(f)

resultado = {k: resultado[k] for k in sorted(resultado)}

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

plt.savefig('figuras/fig_equilibrios_monopolio_2.eps',format='eps')