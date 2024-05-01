import matplotlib.pyplot as plt
import json
import re
import numpy as np

with open('equilibrios_monopolio.json', 'r') as f:
    resultado = json.load(f)

b = np.zeros(int(len(resultado)/2))
idx = 0

b_barra = [re.search("0.\d+",k).group(0) for (k,v) in resultado.items() if '.p' in k]
p_opt = [v for (k,v) in resultado.items() if '.p' in k]

plt.plot(b_barra, p_opt)
plt.show()