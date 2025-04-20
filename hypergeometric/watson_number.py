import sympy as sp
import numpy as np
# Declare symbols
x, n = sp.symbols('x n')
a_val = 5  # You can change a here
n_max = 10

# Define the numerical integral
def numeric_integral(n_val):
    Pn = sp.legendre(n_val, x)
    integrand = (1 - x)**a_val * (1 + x)**a_val * Pn
    return sp.integrate(integrand, (x, -1, 1)).evalf()

# Define the closed-form gamma expression from Whipple's formula
def watson_expression(n_val):
    a = a_val
    #print((1 - n_val) / 2)
    #print(a + 1 - n_val / 2)
    expr = (
        sp.pi* sp.gamma(a + 1)**2 /
        (
            sp.gamma((1 - n_val) / 2) *
            sp.gamma((n_val + 2) / 2) *
            sp.gamma(a + 3/2 + n_val / 2) *
            sp.gamma(a + 1 -n_val / 2)
        )
    )
    return expr.evalf()
'''
def E_asymptotic(n_val):
    a=a_val
    expr=((sp.gamma(a + 1)**2 * sp.gamma(n_val/2 - a)) / (sp.pi * n_val/2 * sp.gamma(a + n_val/2 + 3/2)))
    return expr.evalf()
'''
# Header
print(f"{'n':>2} | {'Numeric Integral':>18} | {'Watson Expression':>20}")
print("-" * 50)

# Loop over even n only
for n_val in np.arange(0, n_max + 1, 0.2):
    I_num = numeric_integral(n_val)
    I_whip = watson_expression(n_val)
    print(f"{n_val:>2} | {float(I_num):>18.6f} | {float(I_whip):>20.6f}")
