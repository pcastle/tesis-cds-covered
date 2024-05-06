import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from sympy import symbols, Piecewise, Min, Max, integrate, pycode, parse_expr, solve, lambdify


# Hacer gráfico de la restricción, el equilibrio, isocuanta
y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)

g = 3*(1-t)*(y-1)**2 + 3*t*y**2

# Defino solo la parte en la que se encuentra el equilibrio

# Valor esperado por cada bono
# p >= b barra & p*q <= b barra
vf = parse_expr((pycode(integrate(Min(p*y/b,1)*g,(y,0,1)))).replace("min(1, b/p)","(b/p)"), locals())

# Theta barra
tbarra = solve(integrate(y*g,(y,0,1)) - b, t)[0]

# Theta gorro
tgorro = solve(vf - p,t)[0]

# Funcion de recaudacion
d = 1 - Max(tbarra,Min(tgorro,1))

# Disminucion de la dimension del problema
q = Min(1,d/p)

u = integrate((Min(p*q/b,1)*y-q)*3*y**2,(y,b/p,1))

q2 = symbols("q2", real = True, positive = True)
u2 = integrate((p*q2/b*y - q2)*3*y**2, (y,b/p,1))



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

b_v = 0.3
resultado = busqueda_equilibrio(b_v,{})


fun_u = solve(u2.subs(b,b_v) - u2.subs([(b,b_v),(p,resultado["0.3.p"]),(q2,recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"])]),q2)[0]

fig, ax = plt.subplots(figsize = [16/2.5,4])


def llega_al_fondo(p):
    return b_v/p if p >= b_v else np.nan



llega_al_fondo_vec = np.vectorize(llega_al_fondo)
isoutilidad1_vec = np.zeros(100)
q_vec = np.zeros(100)

p_v = np.linspace(0,1,100)
for idx, p_vs in enumerate(p_v):
    q_vec[idx] = recaudacion([p_vs],b_v)/p_vs
    q_vec[idx] = q_vec[idx] if q_vec[idx] <= 1 else np.nan
    isoutilidad1_vec[idx] = fun_u.subs(p,p_vs) if p_vs >= resultado["0.3.p"] else np.nan

ax.plot(p_v, llega_al_fondo_vec(p_v), 'k',label = '$q =\\frac{\\bar b}{p}$', )
ax.plot(p_v, q_vec, 'k--',label = '$q(p)$')
ax.plot(p_v,isoutilidad1_vec, 'k:', label = '$u(p,q) = \\bar u$')
ax.plot((resultado["0.3.p"], 1), (recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"], recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"]), 'k:')
ax.scatter(resultado["0.3.p"],recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"])
plt.xticks(np.arange(0, 1, step=0.20))
ax.legend()
# Se añade un grilla
plt.grid(color = '0.95')

# Se añaden etiquetas
ax.set(xlabel = "precio",
       ylabel = "cantidad")

plt.savefig('figuras/equilibrio.eps',format='eps')
# plt.show()


# Hacer gráfico de la restricción, el equilibrio, isocuanta
y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)

g = 3*(1-t*3/4)*(y-1)**2 + 3*t*3/4*y**2

# Defino solo la parte en la que se encuentra el equilibrio

# Valor esperado por cada bono
# p >= b barra & p*q <= b barra
vf = parse_expr((pycode(integrate(Min(p*y/b,1)*g,(y,0,1)))).replace("min(1, b/p)","(b/p)"), locals())

# Theta barra
tbarra = solve(integrate(y*g,(y,0,1)) - b, t)[0]

# Theta gorro
tgorro = solve(vf - p,t)[0]

# Funcion de recaudacion
d = 1 - Max(tbarra,Min(tgorro,1))

# Disminucion de la dimension del problema
q = Min(1,d/p)

u = integrate((Min(p*q/b,1)*y-q)*3*y**2,(y,b/p,1))

q2 = symbols("q2", real = True, positive = True)
u2 = integrate((p*q2/b*y - q2)*3*y**2, (y,b/p,1))



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

b_v = 0.3
resultado = busqueda_equilibrio(b_v,{})


fun_u = solve(u2.subs(b,b_v) - u2.subs([(b,b_v),(p,resultado["0.3.p"]),(q2,recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"])]),q2)[0]

fig, ax = plt.subplots(figsize = [16/2.5,4])


def llega_al_fondo(p):
    return b_v/p if p >= b_v else np.nan



llega_al_fondo_vec = np.vectorize(llega_al_fondo)
isoutilidad1_vec = np.zeros(100)
q_vec = np.zeros(100)

p_v = np.linspace(0,1,100)
for idx, p_vs in enumerate(p_v):
    q_vec[idx] = recaudacion([p_vs],b_v)/p_vs
    q_vec[idx] = q_vec[idx] if q_vec[idx] <= 1 else np.nan
    isoutilidad1_vec[idx] = fun_u.subs(p,p_vs) if p_vs >= resultado["0.3.p"] else np.nan

ax.plot(p_v, llega_al_fondo_vec(p_v), 'k',label = '$q = \\frac{\\bar b}{p}$', )
ax.plot(p_v, q_vec, 'k--',label = '$q(p)$')
ax.plot(p_v,isoutilidad1_vec, 'k:', label = '$u(p,q) = \\bar u$')
ax.plot((resultado["0.3.p"], 1), (recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"], recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"]), 'k:')
ax.scatter(resultado["0.3.p"],recaudacion([resultado["0.3.p"]],b_v)/resultado["0.3.p"])
plt.xticks(np.arange(0, 1, step=0.20))
ax.legend()
# Se añade un grilla
plt.grid(color = '0.95')

# Se añaden etiquetas
ax.set(xlabel = "precio",
       ylabel = "cantidad")

plt.savefig('figuras/equilibrio_2.eps',format='eps')


# Hacer gráfico de la restricción, el equilibrio, isocuanta
y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)

g = 3*(1-t)*(y-1)**2 + 3*t*y**2

# Defino solo la parte en la que se encuentra el equilibrio

# Valor esperado por cada bono
# p >= b barra & p*q <= b barra
vf = parse_expr((pycode(integrate(Min(p*y/b,1)*g,(y,0,1)))).replace("min(1, b/p)","(b/p)"), locals())

# Theta barra
tbarra = solve(integrate(y*g,(y,0,1)) - b, t)[0]

# Theta gorro
tgorro = solve(vf - p,t)[0]

# Funcion de recaudacion
d = 1 - Max(tbarra,Min(tgorro,1))

# Disminucion de la dimension del problema
q = Min(1,d/p)

u = integrate((Min(p*q/b,1)*y-q)*3*y**2,(y,b/p,1))

q2 = symbols("q2", real = True, positive = True)
u2 = integrate((p*q2/b*y - q2)*3*y**2, (y,b/p,1))



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

b_v = 0.4
resultado = busqueda_equilibrio(b_v,{})


fun_u = solve(u2.subs(b,b_v) - u2.subs([(b,b_v),(p,resultado["0.4.p"]),(q2,recaudacion([resultado["0.4.p"]],b_v)/resultado["0.4.p"])]),q2)[0]

fig, ax = plt.subplots(figsize = [16/2.5,4])


def llega_al_fondo(p):
    return b_v/p if p >= b_v else np.nan



llega_al_fondo_vec = np.vectorize(llega_al_fondo)
isoutilidad1_vec = np.zeros(100)
q_vec = np.zeros(100)

p_v = np.linspace(b_v,1,100)
for idx, p_vs in enumerate(p_v):
    q_vec[idx] = recaudacion([p_vs],b_v)/p_vs
    q_vec[idx] = q_vec[idx] if q_vec[idx] <= 1 else np.nan
    isoutilidad1_vec[idx] = fun_u.subs(p,p_vs) if p_vs >= resultado["0.4.p"] else np.nan

ax.plot(p_v, llega_al_fondo_vec(p_v), 'k',label = '$q =\\frac{\\bar b}{p}$', )
ax.plot(p_v, q_vec, 'k--',label = '$q(p)$')
ax.plot(p_v,isoutilidad1_vec, 'k:', label = '$u(p,q) = \\bar u$')
ax.plot((resultado["0.4.p"], 1), (recaudacion([resultado["0.4.p"]],b_v)/resultado["0.4.p"], recaudacion([resultado["0.4.p"]],b_v)/resultado["0.4.p"]), 'k:')
ax.scatter(resultado["0.4.p"],recaudacion([resultado["0.4.p"]],b_v)/resultado["0.4.p"])
plt.xticks(np.arange(0, 1, step=0.20))
ax.legend()
# Se añade un grilla
plt.grid(color = '0.95')

# Se añaden etiquetas
ax.set(xlabel = "precio",
       ylabel = "cantidad")

plt.savefig('figuras/equilibrio_3.eps',format='eps')
# plt.show()


# Hacer gráfico de la restricción, el equilibrio, isocuanta
y, p, q, b = symbols('y p q b', real = True, positive = True)
t = symbols('t', positive = True)

g = 3*(1-t*3/4)*(y-1)**2 + 3*t*3/4*y**2

# Defino solo la parte en la que se encuentra el equilibrio

# Valor esperado por cada bono
# p >= b barra & p*q <= b barra
vf = parse_expr((pycode(integrate(Min(p*y/b,1)*g,(y,0,1)))).replace("min(1, b/p)","(b/p)"), locals())

# Theta barra
tbarra = solve(integrate(y*g,(y,0,1)) - b, t)[0]

# Theta gorro
tgorro = solve(vf - p,t)[0]

# Funcion de recaudacion
d = 1 - Max(tbarra,Min(tgorro,1))

# Disminucion de la dimension del problema
q = Min(1,d/p)

u = integrate((Min(p*q/b,1)*y-q)*3*y**2,(y,b/p,1))

q2 = symbols("q2", real = True, positive = True)
u2 = integrate((p*q2/b*y - q2)*3*y**2, (y,b/p,1))



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

b_v = 0.4
resultado = busqueda_equilibrio(b_v,{})


fun_u = solve(u2.subs(b,b_v) - u2.subs([(b,b_v),(p,resultado["0.4.p"]),(q2,recaudacion([resultado["0.4.p"]],b_v)/resultado["0.4.p"])]),q2)[0]

fig, ax = plt.subplots(figsize = [16/2.5,4])


def llega_al_fondo(p):
    return b_v/p if p >= b_v else np.nan



llega_al_fondo_vec = np.vectorize(llega_al_fondo)
isoutilidad1_vec = np.zeros(100)
q_vec = np.zeros(100)

p_v = np.linspace(b_v,1,100)
p_int = 0
q_int = 0
for idx, p_vs in enumerate(p_v):
    q_vec[idx] = recaudacion([p_vs],b_v)/p_vs
    q_vec[idx] = q_vec[idx] if q_vec[idx] <= 1 else np.nan
    if p_int == 0 and p_vs*q_vec[idx] <= b_v:
        p_int = p_vs
        q_int = q_vec[idx]

    isoutilidad1_vec[idx] = fun_u.subs(p,p_vs) if p_vs*q_vec[idx] <= b_v else np.nan

ax.plot(p_v, llega_al_fondo_vec(p_v), 'k',label = '$q = \\frac{\\bar b}{p}$', )
ax.plot(p_v, q_vec, 'k--',label = '$q(p)$')
ax.plot(p_v,isoutilidad1_vec, 'k:', label = '$u(p,q) = \\bar u$')
ax.plot((p_int, 1), (q_int, q_int), 'k:')
ax.scatter(resultado["0.4.p"],recaudacion([resultado["0.4.p"]],b_v)/resultado["0.4.p"])
plt.xticks(np.arange(0, 1, step=0.20))
ax.legend()
# Se añade un grilla
plt.grid(color = '0.95')

# Se añaden etiquetas
ax.set(xlabel = "precio",
       ylabel = "cantidad")

plt.savefig('figuras/equilibrio_4.eps',format='eps')