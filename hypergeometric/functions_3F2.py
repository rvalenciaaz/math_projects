# -*- coding: utf-8 -*-
"""
hypergeometric_3F2_formulas.sage

Core closed–form evaluations and transformations of the generalized hypergeometric
function ${}_3F_2$ that are classically attributed to Dixon, Watson, Whipple,
Kummer and Thomae.

All routines are symbolic and return Sage expressions, so you can immediately
substitute arbitrary values (numeric or symbolic) and pass the result to
`latex()` for pretty‑printing.

Example usage in a Sage session::

    sage: load('hypergeometric_3F2_formulas.sage')
    sage: a, b, c = var('a b c')
    sage: Dixon(a, b, c)                  # symbolic
    sage: Dixon(1/2, 1/3, 1/4).n()        # numeric
    sage: latex(Watson(a, b, c))          # LaTeX code
"""

from sage.all import hypergeometric, gamma, sqrt, pi, var, latex, SR

###############################################################################
# Basic shortcut for 3F2
###############################################################################

def _3F2(a, b, c, d, e, z=SR(1)):
    """Return the generalized hypergeometric series 3F2([a,b,c];[d,e];z)."""
    return hypergeometric([a, b, c], [d, e], z)

###############################################################################
# 1. Dixon’s well‑poised summation
###############################################################################

def Dixon(a, b, c):
    r"""Classical Dixon summation for a very well‑poised ${}_3F_2(1)$.

    .. math::

        {}_3F_2\!\left(\begin{matrix} a,\; b,\; c \\
        1+a-b,\; 1+a-c \end{matrix}; 1\right)
        = \frac{\Gamma\!\left(\tfrac{1+a}{2}\right)\,
                 \Gamma\!\left(1+\tfrac{a}{2}-b\right)\,
                 \Gamma\!\left(1+\tfrac{a}{2}-c\right)\,
                 \Gamma\!\left(1+\tfrac{a}{2}-b-c\right)}
               {\Gamma(1+a)\,\Gamma(1+a-b)\,\Gamma(1+a-c)\,
                \Gamma\!\left(1+\tfrac{a}{2}-b-c\right)}.
    """
    num = gamma((1 + a) / 2) * gamma(1 + a / 2 - b) * gamma(1 + a / 2 - c) * gamma(1 + a / 2 - b - c)
    den = gamma(1 + a) * gamma(1 + a - b) * gamma(1 + a - c) * gamma(1 + a / 2 - b - c)
    return num / den

###############################################################################
# 2. Watson’s summation
###############################################################################

def Watson(a, b, c):
    r"""Watson’s summation theorem for ${}_3F_2(1)$.

    .. math::

        {}_3F_2\!\left(\begin{matrix} a,\; b,\; c \\
        \tfrac{1}{2}(a+b+1),\; 2c \end{matrix}; 1\right)
        = \frac{\sqrt{\pi}\,\Gamma\!\left(c+\tfrac{1}{2}\right)\,
                 \Gamma\!\left(\tfrac{a+b+1}{2}\right)\,
                 \Gamma\!\left(c-\tfrac{a+b}{2}\right)}
               {\Gamma\!\left(\tfrac{a+1}{2}\right)\,
                 \Gamma\!\left(\tfrac{b+1}{2}\right)\,
                 \Gamma\!\left(c-\tfrac{a}{2}\right)\,
                 \Gamma\!\left(c-\tfrac{b}{2}\right)}.
    """
    num = sqrt(pi) * gamma(c + SR(1) / 2) * gamma((a + b + 1) / 2) * gamma(c - (a + b) / 2)
    den = gamma((a + 1) / 2) * gamma((b + 1) / 2) * gamma(c + (1-a) / 2) * gamma(c + (1-b) / 2)
    return num / den

###############################################################################
# 3. Whipple’s summation
###############################################################################

def Whipple(a, b, c):
    r"""Whipple’s summation theorem for ${}_3F_2(1)$.

    .. math::

        {}_3F_2\!\left(\begin{matrix} a,\; 1+\tfrac{a}{2}-b,\; 1+\tfrac{a}{2}-c \\
        1+a-b,\; 1+a-c \end{matrix}; 1\right)
        = \frac{\Gamma(1+a-b)\,\Gamma(1+a-c)\,
                 \Gamma\!\left(\tfrac{1+a}{2}\right)\,
                 \Gamma\!\left(1+\tfrac{a}{2}-b-c\right)}
               {\Gamma(1+a)\,\Gamma\!\left(\tfrac{1+a}{2}-b\right)\,
                 \Gamma\!\left(\tfrac{1+a}{2}-c\right)\,\Gamma(1+a-b-c)}.
    """
    num = gamma(1 + a - b) * gamma(1 + a - c) * gamma((1 + a) / 2) * gamma(1 + a / 2 - b - c)
    den = gamma(1 + a) * gamma((1 + a) / 2 - b) * gamma((1 + a) / 2 - c) * gamma(1 + a - b - c)
    return num / den

###############################################################################
# 4. Thomae’s two‑term transformation
###############################################################################

def Thomae(a, b, c, d, e):
    r"""Thomae’s fundamental two‑term relation for ${}_3F_2(1)$.

    .. math::

        {}_3F_2\!\left(\begin{matrix} a,\; b,\; c \\
        d,\; e \end{matrix}; 1\right)
        = \frac{\Gamma(d)\,\Gamma(e)\,\Gamma(d+e-a-b-c)}
               {\Gamma(a)\,\Gamma(d+e-a-b)\,\Gamma(d+e-a-c)}\;
          {}_3F_2\!\left(\begin{matrix} d-a,\; e-a,\; d+e-a-b-c \\
          d+e-a-b,\; d+e-a-c \end{matrix}; 1\right).
    """
    pref = gamma(d) * gamma(e) * gamma(d + e - a - b - c) / (
        gamma(a) * gamma(d + e - a - b) * gamma(d + e - a - c)
    )
    return pref * _3F2(d - a, e - a, d + e - a - b - c, d + e - a - b, d + e - a - c)

###############################################################################
# 5. A Kummer‑type quadratic transformation (half‑integer case)
###############################################################################

def Kummer_half(a, b, c):
    r"""A quadratic (Kummer‑type) transformation reducing a half‑integer 3F2(1).

    .. math::

        {}_3F_2\!\left(\begin{matrix} a,\; b,\; c+\tfrac12 \\
        \tfrac12(a+b+1),\; c \end{matrix}; 1\right)
        = \frac{\Gamma(c)\,\Gamma\!\left(c-\tfrac{a+b}{2}\right)}
               {\Gamma\!\left(c-\tfrac{a}{2}\right)\,\Gamma\!\left(c-\tfrac{b}{2}\right)}
          {}_3F_2\!\left(\begin{matrix} \tfrac12 a,\; \tfrac12 b,\; c-\tfrac{a+b}{2} \\
          c-\tfrac{a}{2},\; c-\tfrac{b}{2} \end{matrix}; 1\right).
    """
    pref = gamma(c) * gamma(c - (a + b) / 2) / (gamma(c - a / 2) * gamma(c - b / 2))
    return pref * _3F2(a / 2, b / 2, c - (a + b) / 2, c - a / 2, c - b / 2)

###############################################################################
# Convenience wrappers returning LaTeX code directly
###############################################################################

def dixon_latex(a="a", b="b", c="c"):
    return latex(Dixon(var(a), var(b), var(c)))

def watson_latex(a="a", b="b", c="c"):
    return latex(Watson(var(a), var(b), var(c)))

def whipple_latex(a="a", b="b", c="c"):
    return latex(Whipple(var(a), var(b), var(c)))

def thomae_latex(a="a", b="b", c="c", d="d", e="e"):
    return latex(Thomae(var(a), var(b), var(c), var(d), var(e)))

def kummer_half_latex(a="a", b="b", c="c"):
    return latex(Kummer_half(var(a), var(b), var(c)))
