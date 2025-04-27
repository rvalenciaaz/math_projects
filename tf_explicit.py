import sympy as sp

sp.init_printing()          # pretty console output

def inverse_transformation(n_val: int):
    """
    Return a list [r₁(q), …, rₙ(q)] giving the inverse of the
    Gram–Schmidt construction

        q₀ = (1/√n) Σ rⱼ
        q_k = √(k/(k+1)) ( (1/k) Σ_{j=1..k} rⱼ  −  r_{k+1} ),  k = 1..n−1
    """
    n = sp.Integer(n_val)
    r = sp.IndexedBase('r')
    q = sp.IndexedBase('q')

    # forward map: q_k expressed with the r_j
    r_list = [r[j] for j in range(1, n_val + 1)]
    q_exprs = []

    q0 = (1 / sp.sqrt(n)) * sum(r_list)
    q_exprs.append(q0)

    for k_val in range(1, n_val):
        k = sp.Integer(k_val)
        qk = sp.sqrt(k / (k + 1)) * (
              (1 / k) * sum(r[j] for j in range(1, k_val + 1))
              - r[k_val + 1]
        )
        q_exprs.append(qk)

    # solve {q_k = q_exprs[k]} for the r_j
    sol = sp.solve(
        [sp.Eq(q[i], q_exprs[i]) for i in range(n_val)],
        r_list,
        dict=True
    )[0]

    return [sol[r[j]] for j in range(1, n_val + 1)]


# -------------------------------
# demo
if __name__ == "__main__":
    n_val = 3                      # change this to any n ≥ 2
    inv = inverse_transformation(n_val)
    r = sp.IndexedBase('r')

    for j, expr in enumerate(inv, 1):
        print(f"r_{j} =")
        sp.pprint(expr)
        print()                    # blank line between rows
