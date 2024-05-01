import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the domain
Nx = 300  # Number of grid points
c = 1.0  # Advection velocity
T = 1.5  # Total simulation time
dt = 0.001 # Time step
x = np.linspace(0, L, Nx) # Grid spacing
u_initial = lambda x: np.sin(2*np.pi*x)  # Initial condition

# Discretization
dx = x[1] - x[0]  

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
    for i in range(1, Nx-1):
        fluxes[i] = c * ((u[i+1] - u[i-1])/2) - c**2*((u[i+1] - 2*u[i] + u[i-1])/2)
    fluxes[0] =  c * ((u[1] - u[-1])/2) - c**2*((u[1] - 2*u[0] + u[-1])/2) #periodic bdry cdn given
    fluxes[Nx-1]= c * ((u[0] - u[Nx-2])/2) - c**2*((u[0] - 2*u[Nx-1] + u[Nx-2])/2)
    # Update solution using forward Euler method
    u[0:] -= dt * (fluxes[0:] / dx)
    #u[0] = u[-1]  # Periodic boundary condition

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
