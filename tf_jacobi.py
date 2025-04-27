import sympy as sp

# Setup
sp.init_printing()

# Choose specific n
n_val = 3  # you can change this
n = sp.Integer(n_val)

# Define indexed variables
r = sp.IndexedBase('r')
q = sp.IndexedBase('q')

# Create list of symbols r_1 to r_n
r_list = [r[j] for j in range(1, n_val+1)]
q_list = []

# Define q_0
q0 = (1 / sp.sqrt(n)) * sum(r_list)
q_list.append(q0)

# Define q_k for k=1 to n-1
for k_val in range(1, n_val):
    k = sp.Integer(k_val)
    qk = sp.sqrt(k / (k + 1)) * ( (1/k) * sum(r[j] for j in range(1, k_val+1)) - r[k_val+1] )
    q_list.append(qk)

# Solve for r in terms of q
solutions = sp.solve([sp.Eq(q[i], q_list[i]) for i in range(n_val)], r_list)

# Display the inverse transformation
for idx, expr in enumerate(r_list):
    print(f"r_{idx+1} =")
    sp.pprint(solutions[expr])
    print()
