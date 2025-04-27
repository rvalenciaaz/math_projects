# compare_Y_forms.py
import sympy as sp

def check_identity(n: int):
    """
    Build n symbolic variables X1,…,Xn, construct the two
    expressions for Y_n, simplify their difference and
    return True  ⇔  they are identical.
    """
    if n < 3:
        raise ValueError("Need n ≥ 3")

    # ------------------------------------------------------------------
    # 1. create the sample symbols
    X = sp.symbols(f'X1:{n+1}')          # X1, X2, …, Xn

    # ------------------------------------------------------------------
    # 2. power sums S1,S2,S3
    S1 = sum(X)
    S2 = sum(x**2 for x in X)
    S3 = sum(x**3 for x in X)

    # ------------------------------------------------------------------
    # 3. elementary-symmetric polynomials e1,e2,e3
    e1 = S1
    e2 = sp.Rational(1,2)*(S1**2 - S2)
    e3 = sp.Rational(1,6)*(S1**3 - 3*S1*S2 + 2*S3)

    # original definition of Y_n
    Y_orig = ((n*(e1*e2 - 3*e3) - 2*e1*e2) /
              (n*(e1**2 - 2*e2) - e1**2))

    # ------------------------------------------------------------------
    # 4. centred variables, Q and U
    mean = S1 / n
    d    = [x - mean for x in X]
    Q    = sum(di**2 for di in d)
    U    = sum(di**3 for di in d)

    # new representation
    Y_new = (n-2)/n * S1 - U/Q

    # ------------------------------------------------------------------
    # 5. verify
    return sp.simplify(Y_orig - Y_new) == 0


if __name__ == "__main__":
    for m in (3, 4, 5,6,7,8):
        print(f"n = {m}:  identity holds?  {check_identity(m)}")
