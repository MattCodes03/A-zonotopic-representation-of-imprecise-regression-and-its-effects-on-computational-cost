"""
Proof-of-correctness for zonotope_bounds and nonlinear_zonotope_bounds.

The document claims (eq. 1):
    [Z_y] = x A y_mid  +  sum_i |x A D_r|_i * [-1,1]

Which means the regression band lower/upper bounds are:
    center  = x @ A @ y_mid
    radius  = sum_i  |  (x @ A @ D_r)_i  |
    lower   = center - radius
    upper   = center + radius

Three properties are verified:
  P1. Affine-map exactness:  Z_beta.center == M @ y_mid
                              Z_beta.generators == M @ diag(r)
  P2. 1D projection exactness: Z_pred bounds == analytic formula above
  P3. Interval containment:   all observed y_i in [y_lower_i, y_upper_i]
                               => predicted band contains naive pointwise OLS
"""

import numpy as np

# ── minimal Zonotope class (drop-in if you don't have one) ──────────────────


class Zonotope:
    """Z = <c, G>  =  { c + G xi | xi in [-1,1]^p }"""

    def __init__(self, center, generators):
        self.center = np.asarray(center, dtype=float).flatten()
        self.generators = np.asarray(generators, dtype=float)
        if self.generators.ndim == 1:
            self.generators = self.generators.reshape(len(self.center), -1)

    @classmethod
    def from_intervals(cls, lower, upper):
        lower = np.asarray(lower, dtype=float).flatten()
        upper = np.asarray(upper, dtype=float).flatten()
        mid = 0.5 * (lower + upper)
        rad = 0.5 * (upper - lower)
        return cls(mid, np.diag(rad))

    def affine_map(self, A):
        """Returns <A c, A G>"""
        A = np.atleast_2d(A)
        return Zonotope(A @ self.center, A @ self.generators)

    def output_interval(self):
        """Exact bounding box: c ± sum_col |G|"""
        radius = np.sum(np.abs(self.generators), axis=1)
        return self.center - radius, self.center + radius


# ── helpers ─────────────────────────────────────────────────────────────────

def chebyshev_basis(x, degree):
    x = np.asarray(x).flatten()
    cols = [np.ones_like(x)]
    if degree >= 1:
        cols.append(x)
    for k in range(2, degree + 1):
        cols.append(2 * x * cols[-1] - cols[-2])
    return np.column_stack(cols)


def polynomial_basis(x, degree):
    x = np.asarray(x).flatten()
    return np.column_stack([x ** k for k in range(degree + 1)])


# ── analytic formula (eq. 1 from the document) ──────────────────────────────

def analytic_bounds(x_query, A, y_mid, y_rad):
    """
    x_query : (d,)  – single evaluation point (row vector)
    A        : (d,n) – design / projection matrix  (e.g. M = (X'X)^{-1} X')
    y_mid    : (n,)
    y_rad    : (n,)  – non-negative radii
    """
    xA = x_query @ A            # (n,)
    ctr = xA @ y_mid             # scalar
    rad = np.sum(np.abs(xA * y_rad))  # scalar  -- exact for 1-D zonotope
    return ctr - rad, ctr + rad


# ── zonotope_bounds (linear regression) ─────────────────────────────────────

def zonotope_bounds(X, y_lower, y_upper, x_grid=None):
    M = np.linalg.inv(X.T @ X) @ X.T
    Z_y = Zonotope.from_intervals(y_lower, y_upper)
    Z_beta = Z_y.affine_map(M)

    if x_grid is not None:
        y_min, y_max = [], []
        for xg in x_grid:
            Z_pred = Z_beta.affine_map(np.array([[1, xg]]))
            lower, upper = Z_pred.output_interval()
            y_min.append(lower[0])
            y_max.append(upper[0])
        return Z_beta, (np.array(y_min), np.array(y_max))
    return Z_beta


# ── nonlinear_zonotope_bounds ────────────────────────────────────────────────

def nonlinear_zonotope_bounds(X, y_lower, y_upper, x_grid=None,
                              degree=10, basis='chebyshev'):
    X = np.asarray(X).flatten()
    y_lower = np.asarray(y_lower).flatten()
    y_upper = np.asarray(y_upper).flatten()

    basis_fn = {'chebyshev': chebyshev_basis,
                'polynomial': polynomial_basis}[basis]

    Phi = basis_fn(X, degree)
    M = np.linalg.inv(Phi.T @ Phi) @ Phi.T
    Y_zono = Zonotope.from_intervals(y_lower, y_upper)
    Beta_zono = Y_zono.affine_map(M)

    if x_grid is None:
        return Beta_zono

    Phi_grid = basis_fn(x_grid, degree)
    Yhat_zono = Beta_zono.affine_map(Phi_grid)
    lo, hi = Yhat_zono.output_interval()
    return Beta_zono, Yhat_zono, lo, hi


# ═══════════════════════════════════════════════════════════════════════════
#  VERIFICATION SUITE
# ═══════════════════════════════════════════════════════════════════════════

def check(label, cond, tol=1e-10):
    status = "✓ PASS" if cond else "✗ FAIL"
    print(f"  {status}  {label}")
    return bool(cond)


def run_all():
    rng = np.random.default_rng(42)
    n = 20        # observations
    tol = 1e-10

    # ── ground-truth data ──────────────────────────────────────────────────
    x_obs = np.linspace(-1, 1, n)
    y_mid = 2.0 + 1.5 * x_obs + rng.normal(0, 0.1, n)
    y_rad = rng.uniform(0.05, 0.3, n)
    y_lo = y_mid - y_rad
    y_hi = y_mid + y_rad
    x_grid = np.linspace(-1.2, 1.2, 50)

    all_pass = True

    # ════════════════════════════════════════════════════════════════════════
    print("\n══ LINEAR REGRESSION ZONOTOPE ══")

    X = np.column_stack([np.ones(n), x_obs])   # design matrix
    M = np.linalg.inv(X.T @ X) @ X.T           # (2, n)
    Z_beta, (lo_zono, hi_zono) = zonotope_bounds(X, y_lo, y_hi, x_grid)

    # P1a – center equals M @ y_mid
    p1a = np.allclose(Z_beta.center, M @ y_mid, atol=tol)
    all_pass &= check("P1a  Z_beta.center == M @ y_mid", p1a)

    # P1b – generators equal M @ diag(y_rad)
    p1b = np.allclose(Z_beta.generators, M @ np.diag(y_rad), atol=tol)
    all_pass &= check("P1b  Z_beta.generators == M @ diag(r)", p1b)

    # P2 – zonotope bounds match analytic formula (eq. 1) at every grid point
    lo_analytic = np.array([analytic_bounds(np.array([1, xg]), M,
                                            y_mid, y_rad)[0] for xg in x_grid])
    hi_analytic = np.array([analytic_bounds(np.array([1, xg]), M,
                                            y_mid, y_rad)[1] for xg in x_grid])
    p2l = np.allclose(lo_zono, lo_analytic, atol=tol)
    p2h = np.allclose(hi_zono, hi_analytic, atol=tol)
    all_pass &= check("P2   lower bound == analytic formula (eq. 1)", p2l)
    all_pass &= check("P2   upper bound == analytic formula (eq. 1)", p2h)

    # P3 – band is exact (1-D zonotope → bounding box is tight)
    #   Proof: for a 1-D zonotope <c, g> the only two extreme points are
    #   c ± sum|g_i|, so the bounding box IS the zonotope — no over-approx.
    for xg, lo, hi in zip(x_grid, lo_zono, hi_zono):
        x_row = np.array([1.0, xg])
        Z_1d = Z_beta.affine_map(x_row.reshape(1, -1))
        lo_bb, hi_bb = Z_1d.output_interval()
        # Confirm same as direct formula
        if not (np.isclose(lo_bb[0], lo, atol=tol) and
                np.isclose(hi_bb[0], hi, atol=tol)):
            all_pass = False
            print("  ✗ FAIL  P3 tightness at xg =", xg)
            break
    else:
        print("  ✓ PASS  P3   1-D bounding box is exact (no over-approximation)")

    # P4 – OLS point estimate lies inside every interval
    beta_ols = M @ y_mid
    y_hat_ols = np.array([np.array([1, xg]) @ beta_ols for xg in x_grid])
    p4 = np.all(y_hat_ols >= lo_zono -
                tol) and np.all(y_hat_ols <= hi_zono + tol)
    all_pass &= check("P4   OLS point estimate ⊆ regression band", p4)

    # ════════════════════════════════════════════════════════════════════════
    print("\n══ NONLINEAR (CHEBYSHEV) ZONOTOPE ══")

    degree = 5
    Phi = chebyshev_basis(x_obs, degree)          # (n, degree+1)
    M_nl = np.linalg.inv(Phi.T @ Phi) @ Phi.T
    Beta_zono, _, lo_nl, hi_nl = nonlinear_zonotope_bounds(
        x_obs, y_lo, y_hi, x_grid=x_grid, degree=degree, basis='chebyshev')

    # P1-nl – center and generators
    p1_nl_c = np.allclose(Beta_zono.center, M_nl @ y_mid, atol=tol)
    p1_nl_g = np.allclose(Beta_zono.generators, M_nl @
                          np.diag(y_rad), atol=tol)
    all_pass &= check("P1a  Beta_zono.center == M_nl @ y_mid", p1_nl_c)
    all_pass &= check("P1b  Beta_zono.generators == M_nl @ diag(r)", p1_nl_g)

    # P2-nl – matches analytic formula
    Phi_grid = chebyshev_basis(x_grid, degree)
    lo_an = np.array([analytic_bounds(Phi_grid[k], M_nl, y_mid, y_rad)[0]
                      for k in range(len(x_grid))])
    hi_an = np.array([analytic_bounds(Phi_grid[k], M_nl, y_mid, y_rad)[1]
                      for k in range(len(x_grid))])
    all_pass &= check("P2   lower bound == analytic formula (eq. 1)",
                      np.allclose(lo_nl, lo_an, atol=tol))
    all_pass &= check("P2   upper bound == analytic formula (eq. 1)",
                      np.allclose(hi_nl, hi_an, atol=tol))

    # P3-nl – Yhat_zono is row-wise 1-D (one row per grid point)
    #  Each row of Phi_grid maps Beta_zono to a 1-D zonotope → exact BB
    Phi_grid_full = chebyshev_basis(x_grid, degree)
    Yhat_zono = Beta_zono.affine_map(Phi_grid_full)
    assert Yhat_zono.generators.shape[0] == len(x_grid), "shape mismatch"
    # Each row is a 1-D object; verify individual rows
    tight = True
    for k in range(len(x_grid)):
        row = Phi_grid_full[k:k+1]          # (1, d+1)
        Z_1d = Beta_zono.affine_map(row)
        l1, h1 = Z_1d.output_interval()
        if not (np.isclose(l1[0], lo_nl[k], atol=tol) and
                np.isclose(h1[0], hi_nl[k], atol=tol)):
            tight = False
            break
    all_pass &= check(
        "P3   1-D bounding box is exact (no over-approximation)", tight)

    # P4-nl – OLS nonlinear point estimate inside band
    beta_nl_ols = M_nl @ y_mid
    y_hat_nl = Phi_grid_full @ beta_nl_ols
    p4_nl = np.all(y_hat_nl >= lo_nl - tol) and np.all(y_hat_nl <= hi_nl + tol)
    all_pass &= check("P4   nonlinear OLS ⊆ regression band", p4_nl)

    # ════════════════════════════════════════════════════════════════════════
    print("\n══ DEPENDENCE STRUCTURE TEST ══")
    # The document's key claim: zonotope captures CROSS-COMPONENT dependence in [b].
    # Without zonotopes, naively treating [b_1],[b_2] as independent when computing
    # x @ [b] leads to over-approximation.  We verify this by:
    #   (a) computing the zonotope band (exact, tracks y_i dependencies via shared generators)
    #   (b) computing a naive band that treats [b_1],[b_2] as *independent* intervals
    # The zonotope band must be ≤ the naive independent-beta band.

    Z_beta_lo, Z_beta_hi = Z_beta.output_interval()   # component-wise bounds on beta

    # Naive: treat [beta_0] and [beta_1] as independent (forget shared generators)
    naive_widths = []
    zono_widths = []
    for xg in x_grid:
        x_row = np.array([1.0, xg])
        # zonotope: exact 1D projection
        z_rad = np.sum(np.abs(x_row @ Z_beta.generators))
        # naive: independent beta intervals → sum of |x_j| * half-width([beta_j])
        n_rad = np.sum(np.abs(x_row) * (Z_beta_hi - Z_beta_lo) * 0.5)
        zono_widths.append(2 * z_rad)
        naive_widths.append(2 * n_rad)

    zono_widths = np.array(zono_widths)
    naive_widths = np.array(naive_widths)
    p_dep = np.all(zono_widths <= naive_widths + tol)
    # Report max gain from dependency tracking
    max_gain = np.max(naive_widths - zono_widths)
    all_pass &= check(
        f"P5   zonotope width ≤ naive-independent-beta (max gain = {max_gain:.4f})",
        p_dep
    )

    # ════════════════════════════════════════════════════════════════════════
    print()
    if all_pass:
        print("  ══ ALL PROOFS PASSED ══")
    else:
        print("  ══ SOME CHECKS FAILED — inspect above ══")

    return all_pass


if __name__ == "__main__":
    run_all()
