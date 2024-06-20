import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import argparse
from numfluxes import *

def smooth(x):
    return np.sin(2*np.pi*x)
def equation(u, x, t):
    return u - np.sin(2*np.pi*(x - u*t))

# Initial condition
u_initial = lambda x: np.sin(2*np.pi*x)  # u(x, 0) = sin(2*pi*x)

def solve(N, cfl, scheme, Tf, uinit):
    xmin, xmax = 0.0, 1.0

    x = np.linspace(xmin, xmax, N)
    h = (xmax - xmin) / (N - 1)
    u = uinit(x)
    dt = cfl * h / np.max(u)
    lam = dt / h

    t, it = 0.0, 0
    while t < Tf:
        if scheme == 'C':
            f = flux_central(lam, u)
        elif scheme == 'LF':
            f = flux_lf(lam, u)
        elif scheme == 'GLF':
            f = flux_glf(u)
        elif scheme == 'LLF':
            f = flux_llf(u)
        elif scheme == 'LW':
            f = flux_lw(lam, u)
        elif scheme == 'ROE':
            f = flux_roe(u)
        elif scheme == 'EROE':
            f = flux_eroe(u)
        elif scheme == 'GOD':
            f = flux_god(u)
        else:
            print("Unknown scheme: ", scheme)
            return
        u[1:-1] -= lam * (f[2:-1] - f[1:-2])
        t += dt
        it += 1
    u_values = []
    for x_val in x:
        zero_of_u = fsolve(equation, u_initial(x_val), args=(x_val, t))
        u_values.append(zero_of_u[0])  # Store only the zeroth element of the array

    print("Zeroes of u(x, t) at t=0.15:")
    for i, x_val in enumerate(x):
        print(f"x = {x_val:.3f}, u = {u_values[i]:.21f}")

    # Compute errors
    err = np.abs(u - u_values)
    em = np.max(err)
    e1 = h * err[0] + h * np.sum(err[1:-1])
    err = err ** 2
    e2 = np.sqrt(h * err[0] + h * np.sum(err[1:-1]))
    return em, e1, e2

    print("Zeroes of u(x, t) at t=0.15:")
    for i, x_val in enumerate(x):
        print(f"x = {x_val:.3f}, u = {u_values[i]:.21f}")

    # Compute errors
    #em, e1, e2 = solve(len(x), 0.5, 'C', t, u_initial)

    print("Time:", t)
    print("Max error (em):", em)
    print("Error L1 (e1):", e1)
    print("Error L2 (e2):", e2)

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-N', metavar='N', type=int, nargs='+', help='Number of cells', required=True, default=[50,100,200,400])
parser.add_argument('-cfl', type=float, help='CFL number', default=0.9)
parser.add_argument('-scheme', choices=('C','LF','GLF','LLF','LW','ROE','EROE','GOD'), help='Scheme', default='LF')
parser.add_argument('-Tf', type=float, help='Final time', default=0.10)
args = parser.parse_args()

# Run the solver for different number of grid points
emax, e1, e2 = np.empty(len(args.N)), np.empty(len(args.N)), np.empty(len(args.N))
i = 0
for N in args.N:
    print("Running for cells = ", N)
    emax[i], e1[i], e2[i] = solve(N, args.cfl, args.scheme, args.Tf, smooth)
    i += 1

# Compute convergence rate
for i in range(1, len(emax)):
    pmax = np.log(emax[i-1]/emax[i]) / np.log(2.0)
    p1 = np.log(e1[i-1]/e1[i]) / np.log(2.0)
    p2 = np.log(e2[i-1]/e2[i]) / np.log(2.0)
    print(f'For N={args.N[i]}: L1error ={e1[i]}, L2error ={e2[i]}, Linferror={emax[i]}')
    print(f'Convergence rates: L1={p1}, L2={p2}, Linf={pmax}')

# Plot error convergence
plt.loglog(args.N, e1, '*-', label='$L_1$ norm')
plt.loglog(args.N, e2, 's-', label='$L_2$ norm')
plt.loglog(args.N, emax, 'o-', label='$L_\infty$ norm')
x_ref = np.linspace(args.N[0], args.N[-1], 100)
y_ref = 2*(x_ref / args.N[0]) ** -1 * e1[0]  # Assuming e1[0] for reference
plt.loglog(x_ref, y_ref, '--', color='gray', label='line with lope 1')
plt.xlabel('Number of cells')
plt.ylabel('Error norm')
plt.title('Scheme=' + args.scheme + ', CFL=' + str(args.cfl))
plt.legend()
plt.grid(True)
plt.show()
