import matplotlib.pyplot as plt
import sympy 
from monopolio import *
import numpy as np

x = np.linspace(0.25,1,100)
print(d.subs([(b,0.25),(p,0.7)])/0.7)
print(u.subs([(b,0.25),(p,0.7)]))

sympy.plot(u.subs(b,0.62), (p,0.62,1))



