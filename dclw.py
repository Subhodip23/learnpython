import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10.0  # Length of the domain
Nx = 300  # Number of spatial points
T = 2.0   # Total simulation time
Nt = 100  # Number of time steps
dt = T / Nt  # Time step size
dx = L / Nx  # Spatial step size
c = 1.0    # Advection speed

# Initial condition
def initial_condition(x):
    return np.where(np.logical_and(x >= 4.0, x <= 6.0), 1.0, 0.0)

# Exact solution
def exact_solution(x, t):
    return initial_condition(x - c * t)

# Spatial grid
x = np.linspace(0, L, Nx)
u = initial_condition(x)

# Plot initial condition
plt.plot(x, u, label='t = 0')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Initial Condition')
plt.grid(True)
plt.legend()
plt.show()

# Time stepping and plotting
t = 0
while t < T:
    u_new = np.zeros_like(u)
    for i in range(1, Nx - 1):
        u_new[i] = u[i] - 0.5 * c * (dt / dx) * (u[i+1] - u[i-1]) + 0.5 * (c * dt / dx)**2 * (u[i+1] - 2 * u[i] + u[i-1])
    u_new[0] = u[0] - 0.5 * c * (dt / dx) * (u[1] - u[-1]) + 0.5 * (c * dt / dx)**2 * (u[1] - 2 * u[0] + u[-1])
    u_new[-1] = u_new[0]  # Periodic boundary condition
    u = u_new.copy()
    
    # Exact solution
    u_exact = exact_solution(x, t)
    
    t += dt
    plt.plot(x, u, label='Lax-Wendroff Numerical')
    plt.plot(x, u_exact, label='Exact', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Solution at t = {t:.2f}')
    plt.grid(True)
    plt.legend()
    plt.pause(0.1)  # Pause to see the plot
    plt.clf()

