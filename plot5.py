import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def equation(u, x, t):
    return u - np.sin(x - u*t)

# Initial condition
u_initial = lambda x: np.sin(x)

# Time parameters
t = 0
dt = 0.01
t_max = 1.01  # Example maximum time

x_values = np.linspace(0, 2*np.pi, 100)

u_all_values = []  # List to store u_values for all t values

plt.figure(figsize=(8, 6))
while t <= t_max:
    u_values = []
    for x_val in x_values:
        zero_of_u = fsolve(equation, u_initial(x_val), args=(x_val, t))
        u_values.append(zero_of_u)

    u_values = np.array(u_values).reshape(-1, len(x_values))
    
    # Append u_values for the current t value to u_all_values
    u_all_values.append(u_values)
   
    # Plot the solution for the current time step
    plt.clf()  # Clear the previous plot
    for i in range(len(u_values)):
        plt.plot(x_values, u_values[i], label=f"t = {t:.2f}")

    plt.title('Solution plot of of u(x, t) = sin(x - u(x,t)t')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)

    t += dt

# Concatenate u_values for all t values vertically
u_all_values = np.concatenate(u_all_values, axis=0)

# Save u_all_values to a text file
np.savetxt('u_values.txt', u_all_values)
plt.show() 

