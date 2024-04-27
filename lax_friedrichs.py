import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the domain
nx = 100  # Number of grid points in space
dx = L / nx  # Grid spacing
nt = 100  # Number of time steps
dt = 0.01  # Time step size
c = 1.0  # Velocity

# Initialize solution array
u = np.zeros(nx)

# Initial condition
u[int(0.5 / dx):int(0.6 / dx)] = 1.0  # Square wave initial condition

# Time integration using Lax-Friedrichs finite volume method
for n in range(nt):
    u_new = np.zeros(nx)
    for i in range(1, nx - 1):
        flux_left = 0.5 * c * (u[i] + u[i - 1]) - 0.5 * dx / dt * (u[i] - u[i - 1])
        flux_right = 0.5 * c * (u[i + 1] + u[i]) - 0.5 * dx / dt * (u[i + 1] - u[i])
        u_new[i] = u[i] - (dt / dx) * (flux_right - flux_left)
    u = u_new.copy()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, L, nx), u, label=f"t = {nt * dt:.2f}")
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of Linear Transport Equation using Lax-Friedrichs Finite Volume Method')
plt.legend()
plt.grid(True)
plt.show()

