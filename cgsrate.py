import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    return np.sin(2 * np.pi * x)

def exact_solution(x, t):
    return initial_condition(x - t)

def upwind_scheme(u_old, dx, dt, c):
    u_new = np.zeros_like(u_old)
    u_new[1:] = u_old[1:] - c * dt / dx * (u_old[1:] - u_old[:-1])
    u_new[0] = u_old[0] - c * dt / dx * (u_old[0] - u_old[-1]) # Periodic boundary condition
    return u_new

def calculate_error(u_numerical, u_exact):
    error_l1 = np.sum(np.abs(u_numerical - u_exact)) / len(u_exact)
    error_l2 = np.sqrt(np.sum((u_numerical - u_exact)**2) / len(u_exact))
    error_linfinity = np.max(np.abs(u_numerical - u_exact))
    return error_l1, error_l2, error_linfinity

def convergence_rate(errors, refinements):
    rates = []
    for i in range(len(errors) - 1):
        rate = np.log(errors[i] / errors[i + 1]) / np.log(refinements[i] / refinements[i + 1])
        rates.append(rate)
    return rates

# Parameters
L = 1.0  # Length of the domain
T = 1.0  # Final time
c = 1.0  # Advection speed
num_points = [50, 100, 200, 400]  # Different grid sizes
dx_values = [L / n for n in num_points]
dt_values = [0.5 * dx for dx in dx_values]

errors_l1 = []
errors_l2 = []
errors_linfinity = []

for dx, dt in zip(dx_values, dt_values):
    nx = int(L / dx) + 1
    nt = int(T / dt) + 1

    x = np.linspace(0, L, nx)
    u_initial = initial_condition(x)
    u_exact_final = exact_solution(x, T)

    u_numerical = np.copy(u_initial)
    for t in range(nt):
        u_numerical = upwind_scheme(u_numerical, dx, dt, c)

    error_l1, error_l2, error_linfinity = calculate_error(u_numerical, u_exact_final)
    errors_l1.append(error_l1)
    errors_l2.append(error_l2)
    errors_linfinity.append(error_linfinity)

# Calculate convergence rates
refinements = np.array(dx_values)
rates_l1 = convergence_rate(errors_l1, refinements)
rates_l2 = convergence_rate(errors_l2, refinements)
rates_linfinity = convergence_rate(errors_linfinity, refinements)

# Print convergence rates
print("Convergence Rates:")
print("L1 norm:", rates_l1)
print("L2 norm:", rates_l2)
print("L-infinity norm:", rates_linfinity)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(refinements[:-1], errors_l1[:-1], marker='o', linestyle='-', label='L1 norm')
plt.loglog(refinements[:-1], errors_l2[:-1], marker='s', linestyle='-', label='L2 norm')
plt.loglog(refinements[:-1], errors_linfinity[:-1], marker='^', linestyle='-', label='L-infinity norm')
plt.xlabel('Grid Spacing (dx)')
plt.ylabel('Error')
plt.title('Convergence Plot')
plt.legend()
plt.grid(True)
plt.show()

