import matplotlib.pyplot as plt
import json
import re

with open('equilibrios_monopolio.json', 'r') as f:
    resultado = json.load(f)

b = len(resultado)/2
for x,y in resultado.items():
    
    b = re.search("0.\d+",x).group(0)

