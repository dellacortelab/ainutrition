
from sympy import symbols, Eq, Function, integrate, dsolve, lambdify

# Define the variables
t = symbols('t', real=True)
usa, k10 = symbols('usa k10', real=True)
MGgut = Function('MGgut')(t)
Gpl = Function('Gpl')(t)
Ipl = Function('Ipl')(t)
Usc1 = Function('Usc1')(t)
Usc2 = Function('Usc2')(t)
Irem_f = Function('Irem_f')(t)  # Use a different symbol for the function

# Define the parameters
sigma, k1, Dmeal, k2, gbliv, Gbpl, k4, beta, Irem, f, vG, Mb, KM, c1, Gthpl, k5, tau_i, tau_d, h, t_half, vI, Ula, k6, k7, t_int, k8, k9, k10, Ibpl, k11, k12 = symbols('sigma k1 Dmeal k2 gbliv Gbpl k4 beta Irem f vG Mb KM c1 Gthpl k5 tau_i tau_d h t_half vI Ula k6 k7 t_int k8 k9 k10 Ibpl k11 k12')

# Define the constant values
k1 = 1.45e-2
k2 = 2.76e-1
k3 = 6.07e-3
k4 = 2.35e-4
k5 = 9.49e-2
k6 = 1.93e-1
k7 = 1.15
k8 = 7.27
k9 = 0
k10 = 0
k11 = 3.83e-2
k12 = 2.84e-1
k13 = 0
sigma = 1.34
K_M = 13.2
gbliv = .043
Gthpl = 9
vG = 17/70
vI = 13/70
beta = 1
f = .005551
tau_i = 31
t_int = 30
tau_d = 3
c1 = .1

# Define the differential equations
dMGgut_dt = Eq(MGgut.diff(t), sigma * k1**(sigma) * t**(sigma-1) * (-(k1*t)**(sigma)) * Dmeal - k2 * MGgut)
dGpl_dt = Eq(Gpl.diff(t), (gbliv*(Gpl - Gbpl) - k4 * beta * Irem_f * Gpl + (f / (vG * Mb)) * k2 * MGgut - gbliv*((KM + Gbpl)/ Gbpl)*(Gpl / (KM + Gpl)) - k5 * beta * Irem_f * (Gpl / (KM + Gpl)) - (c1 / (vG * Mb)) * (Gpl - Gthpl)).simplify())
dIpl_dt = Eq(Ipl.diff(t), ((1 / beta) * (k6 * (Gpl - Gbpl) + (k7 / tau_i) * integrate((Gpl - Gbpl), (t, t, t_int)) + (k7 / tau_i) * Gbpl + k8 * tau_d * Gpl.diff(t))) + k9 * (1 / (vI * Mb)) * Usc2 + ((h * t_half**h * t**(h - 1)) / ((t_half**h + t**h)**2)) * ((1 / (vI * Mb)) * Ula) - k7 * (Gbpl / (beta * tau_i * Ibpl)) * Ipl)
dUsc1_dt = Eq(Usc1.diff(t), usa - k10 * Usc1)
dUsc2_dt = Eq(Usc2.diff(t), k10 * Usc1 - k9 * Usc2)
dIrem_dt = Eq(Irem_f.diff(t), k11 * (Ipl - Ibpl) - k12 * Irem_f)

# Solve the differential equations
solution_MGgut = dsolve(dMGgut_dt, MGgut)
solution_Gpl = dsolve(dGpl_dt, Gpl)
solution_Ipl = dsolve(dIpl_dt, Ipl)
solution_Usc1 = dsolve(dUsc1_dt, Usc1)
solution_Usc2 = dsolve(dUsc2_dt, Usc2)
solution_Irem = dsolve(dIrem_dt, Irem_f)

# Convert the solutions to Python functions
MGgut_func = lambdify(t, solution_MGgut.rhs)
Gpl_func = lambdify(t, solution_Gpl.rhs)
Ipl_func = lambdify(t, solution_Ipl.rhs)
Usc1_func = lambdify(t, solution_Usc1.rhs)
Usc2_func = lambdify(t, solution_Usc2.rhs)
Irem_func = lambdify(t, solution_Irem.rhs)

# Test the functions by evaluating them at a specific time
import numpy as np
import matplotlib.pyplot as plt

# Test the functions by evaluating them at a specific time
t_val = 1.5  # Specify the time value
MGgut_val = MGgut_func(t_val)
Gpl_val = Gpl_func(t_val)
Ipl_val = Ipl_func(t_val)
Usc1_val = Usc1_func(t_val)
Usc2_val = Usc2_func(t_val)
Irem_val = Irem_func(t_val)

# Print the results
print("MGgut(t) =", MGgut_val)
print("Gpl(t) =", Gpl_val)
print("Ipl(t) =", Ipl_val)
print("Usc1(t) =", Usc1_val)
print("Usc2(t) =", Usc2_val)
print("Irem(t) =", Irem_val)

# Specify the time range
t_start = 0
t_end = 10
num_points = 1000
t_values = np.linspace(t_start, t_end, num_points)

# Evaluate the functions at the specified time range
MGgut_values = MGgut_func(t_values)
Gpl_values = Gpl_func(t_values)
Ipl_values = Ipl_func(t_values)
Usc1_values = Usc1_func(t_values)
Usc2_values = Usc2_func(t_values)
Irem_values = Irem_func(t_values)

# Plot the graphs
plt.figure(figsize=(10, 6))
plt.plot(t_values, MGgut_values, label='MGgut(t)')
plt.plot(t_values, Gpl_values, label='Gpl(t)')
plt.plot(t_values, Ipl_values, label='Ipl(t)')
plt.plot(t_values, Usc1_values, label='Usc1(t)')
plt.plot(t_values, Usc2_values, label='Usc2(t)')
plt.plot(t_values, Irem_values, label='Irem(t)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Differential Equations')
plt.legend()
plt.grid(True)
plt.show()
