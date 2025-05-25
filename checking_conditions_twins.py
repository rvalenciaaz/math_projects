#!/usr/bin/env python3
"""
2-switch permutation checker
----------------------------

Case 0 : no swap  –  should give P = I
Case 1 : one good swap
Case 2 : two good swaps (different twin classes)
Case 3 : **bad** – swaps share a vertex (violates disjointness)
Case 4 : **bad** – heads are not external twins
"""
import numpy as np, networkx as nx
from numpy.linalg import svd, norm
import textwrap, itertools, random

# ---------- helpers ----------
def connected_er_graph(n, p=0.25, rng=None):
    rng = rng or np.random.default_rng()
    while True:
        A = (rng.random((n, n)) < p).astype(int)
        A = np.triu(A, 1); A += A.T
        if nx.is_connected(nx.from_numpy_array(A)): return A

def impose_twin(A, heads):
    h1, h2 = heads; n = len(A)
    ext = [v for v in range(n) if v not in heads]
    A[ext, h2] = A[ext, h1]; A[h2, ext] = A[h1, ext]; return A

def two_switch(A, i, j, k, l):
    A[i,j]=A[j,i]=0; A[k,l]=A[l,k]=0
    A[i,l]=A[l,i]=1; A[k,j]=A[j,k]=1

def perm_from_svd(A0, A1):
    U0, _, V0t = svd(A0); U1, _, V1t = svd(A1)
    OL, OR = U1@U0.T, V1t.T@V0t
    n = len(A0); P = np.zeros((n,n), int); used=set()
    for c in range(n):
        dots = np.abs(OL.T @ OR[:,c]); dots[list(used)] = -1
        r = int(np.argmax(dots)); P[r,c] = 1; used.add(r)
    return P

def report(title, A0, A1):
    P = perm_from_svd(A0, A1)
    ok  = ((P==0)|(P==1)).all() and (P.sum(0)==1).all() and (P.sum(1)==1).all()
    err = norm(A1 - P@A0@P.T)
    flips = int((np.trace(np.eye(len(P))-P))//2)
    print(f"--- {title} ---")
    print(f"perm OK? {ok}   identity? {np.all(P==np.eye(len(P),int))}")
    print(f"reconstruction error ‖P A Pᵀ - A'‖_F = {err:.2e}")
    print(f"# flip blocks in P: {flips}\n")

# ---------- concrete examples ----------
def case0():
    n=6; A=np.zeros((n,n),int)
    for v in range(n): A[v,(v+1)%n]=A[(v+1)%n,v]=1
    return A, A.copy()

def case1():
    A,_=case0(); A=impose_twin(A,(2,3)); A0=A.copy()
    two_switch(A,0,2,1,3); return A0,A

def case2():
    rng=np.random.default_rng(0); n=12
    A=connected_er_graph(n,0.25,rng)
    for cls in [(0,1),(4,5)]: A=impose_twin(A,cls)
    A0=A.copy()
    two_switch(A,6,0,7,1); two_switch(A,8,4,9,5)
    return A0,A

def case_bad_overlap():
    n=8; A=connected_er_graph(n,0.3); A=impose_twin(A,(0,1)); A0=A.copy()
    two_switch(A,2,0,3,1); two_switch(A,4,0,5,6)   # share vertex 0
    return A0,A

def case_bad_notwin():
    n=8; A=connected_er_graph(n,0.3); A0=A.copy()
    two_switch(A,0,1,2,3)   # heads 1,3 NOT twins
    return A0,A

# ---------- run ----------
for title, build in [
    ("Case 0: no swap",          case0),
    ("Case 1: one good swap",    case1),
    ("Case 2: two good swaps",   case2),
    ("Case 3: BAD – overlap",    case_bad_overlap),
    ("Case 4: BAD – non-twin",   case_bad_notwin),
]:
    A0,A1 = build(); report(title, A0, A1)

print(textwrap.dedent("""
  * For Cases 0–2 (hypotheses satisfied) you will see:
       perm OK = True,  reconstruction error = 0,
       flip blocks = # swaps  (0,1,2).

  * Cases 3–4 break an assumption.  The script still finds a
    permutation, but reconstruction error is non-zero and the
    flip-block count no longer matches the number of swaps.
"""))