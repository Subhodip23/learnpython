import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the domain
Nx = 100  # Number of grid points
c = 1.0  # Advection velocity
T = 1.0  # Total simulation time
dt = 0.001  # Time step (reduced)
u_initial = lambda x: np.sin(2*np.pi*x)  # Initial condition

# Discretization
x = np.linspace(0, L, Nx)  # Grid points
dx = x[1] - x[0]  # Grid spacing

# Initialize solution array
u = u_initial(x)

# Plot initial condition
plt.figure()
plt.plot(x, u, label='Initial Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of Advection Equation (Lax-Friedrichs)')
plt.legend()

# Main time-stepping loop
t = 0.0
while t < T:
    # Compute fluxes using Lax-Friedrichs scheme
    fluxes = np.zeros(Nx)
    for i in range(1, Nx-1):
        fluxes[i] = 0.5 * (u[i+1] + u[i-1]) - 0.5 * c * dt / dx * (u[i+1] - u[i-1])

    # Update solution using Lax-Friedrichs scheme
    u[1:-1] = 0.5 * (u[:-2] + u[2:]) - 0.5 * c * dt / dx * (u[2:] - u[:-2])

    # Apply periodic boundary conditions
    u[0] = u[-2]
    u[-1] = u[1]

    # Update time
    t += dt

    # Update plot
    plt.cla()  # Clear the current plot
    plt.plot(x, u, label='Current Solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution of Advection Equation (Lax-Friedrichs) at t = {:.2f}'.format(t))
    plt.legend()
    plt.pause(0.01)  # Pause to allow for visualization

plt.show()
