import sympy as sp
sp.init_printing()                       # pretty output

# ------------------------------------------------------------
# closed-form coefficients of the inverse Gram–Schmidt map
#   r₁  = 1/√n q₀  +  Σ_{k=1..n-1} 1/√(k(k+1)) q_k
#   rᵢ  = 1/√n q₀  −  √((i−1)/i) q_{i−1}
#                   +  Σ_{k=i..n-1} 1/√(k(k+1)) q_k ,   i=2…n
# ------------------------------------------------------------
def inverse_direct(n_val: int):
    n      = sp.Integer(n_val)
    q      = sp.IndexedBase('q')
    result = []

    for i in range(1, n_val + 1):
        terms = [q[0] / sp.sqrt(n)]                 # common barycentric part

        if i >= 2:                                  # the “minus” leg
            I = sp.Integer(i)
            terms.append(sp.sqrt((I - 1) / I) * q[i - 1])

        # “tail” coefficients  1/√(k(k+1))
        for k in range(max(1, i), n_val):
            K = sp.Integer(k)
            terms.append(-q[k] / sp.sqrt(K * (K + 1)))

        result.append(sp.simplify(sp.Add(*terms)))

    return result


def pretty_print_inverse(n_val: int = 4):
    r_exprs = inverse_direct(n_val)
    for j, expr in enumerate(r_exprs, 1):
        print(f"r_{j} =")
        sp.pprint(expr)
        print()                    # blank line between rows


# ---------------- demo ----------------
if __name__ == "__main__":
    pretty_print_inverse(n_val=4)   # change this to any n ≥ 2
