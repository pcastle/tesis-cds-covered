import numpy as np
from mayavi import mlab
from scipy.optimize import fsolve

# Define the functions
def f(x, y):
    return x**2 + y**2

def g(x, y):
    return 2*x**2 + y

def h(x, y):
    return x + y**2

# Create a mesh grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Compute the function values
Z1 = f(X, Y)
Z2 = g(X, Y)
Z3 = h(X, Y)

# Plot the first function
mlab.figure(bgcolor=(1, 1, 1))
mlab.mesh(X, Y, Z1, colormap='Blues', opacity=0.5)

# Plot the second function
mlab.mesh(X, Y, Z2, colormap='Reds', opacity=0.5)

# Plot the third function
mlab.mesh(X, Y, Z3, colormap='Greens', opacity=0.5)

# Define the system of equations to find the intersection points
def equations(vars):
    x, y, z = vars
    eq1 = f(x, y) - z
    eq2 = g(x, y) - z
    eq3 = h(x, y) - z
    return [eq1, eq2, eq3]

# Initial guess for the variables
initial_guess = [1, 1, 2]

# Solve the system of equations
solution = fsolve(equations, initial_guess)
print(f"Intersection point: {solution}")

# Extract the intersection point
x_int, y_int, z_int = solution

# Plot the intersection point
mlab.points3d(x_int, y_int, z_int, color=(0, 0, 0), scale_factor=0.3)



mlab.show()
