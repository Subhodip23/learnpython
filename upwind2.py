import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10.0   # Length of the domain
T = 5.0    # Total time
Nx = 100   # Number of spatial grid points
dx = L / Nx  # Spatial step size
dt = 0.01  # Time step size
c = 1.0    # Advection SPEED

# Initial condition
def initial_condition(x):
    return np.sin(2 * np.pi * x / L)

# Grid
x = np.linspace(0, L, Nx, endpoint=False)  # Exclude endpoint for periodicity
u = initial_condition(x)

# Main time-stepping loop
t = 0.0
while t < T:
    # Compute fluxes using upwind scheme
    fluxes = np.zeros(Nx)
    for i in range(Nx):
        # Upwind scheme with periodic boundary condition
        fluxes[i] = c * (u[i] - u[(i - 1) % Nx])

    # Update solution using forward Euler method
    u -= dt * (fluxes / dx)

    # Update time
    t += dt

    # Update plot
    plt.cla()  # Clear the current plot
    plt.plot(x, u, label='Current Solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution of Advection Equation (Upwind Scheme) at t = {:.2f}'.format(t))
    plt.legend()
    plt.pause(0.01)  # Pause to allow for visualization

plt.show()

