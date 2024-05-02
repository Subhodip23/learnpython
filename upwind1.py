import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the domain
Nx = 300  # Number of grid points
c = 1.0  # Advection velocity
T = 2.0  # Total simulation time
dt = 0.001  # Time step
u_initial = lambda x: np.sin(2*np.pi*x)  # Initial condition

# Discretization
x = np.linspace(0, L, Nx, endpoint=False)  # Grid points
dx = x[1] - x[0]  # Grid spacing

# Initialize solution array
u = u_initial(x)

# Plot initial condition
plt.figure()
plt.plot(x, u, label='Initial Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of Advection Equation (Finite Difference)')
plt.legend()

# Main time-stepping loop
t = 0.0
while t < T:
    # Compute fluxes using upwind scheme
    fluxes = np.zeros(Nx)
    for i in range(1, Nx):
        fluxes[i] = c * (u[i] - u[i-1])

    # Update solution using forward Euler method
    u[1:] -= dt * (fluxes[1:] / dx)
    u[0] = u[-1]  # Periodic boundary condition

    # Update time
    t += dt

    # Update plot
    plt.cla()  # Clear the current plot
    plt.plot(x, u, label='Current Solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution of Advection Equation (Finite Difference) at t = {:.2f}'.format(t))
    plt.legend()
    plt.pause(0.01)  # Pause to allow for visualization

plt.show()