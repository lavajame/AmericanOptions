# Andersen (Fast Americans with Integral Methods) — integration plan for this repo

This note is a working implementation plan to add a *second* American pricer based on the Andersen “European + Early Exercise Premium (EEP)” representation, while keeping the existing COS rollback pricer intact for validation.

## Key math to implement

### Main representation (Eq. 11)
From the paper (page 25), for an American **put** with exercise boundary $S^*_T(t)$, for $S \ge S^*_T(t)$:

\[
P(t,S) = p(t,S)
+ rK \int_t^T e^{-r(u-t)}\, \mathbb{E}[\mathbf{1}\{S(u) < S^*_T(u)\} \mid S(t)=S] \, du
- (r-\mu) \int_t^T e^{-r(u-t)}\, \mathbb{E}[S(u)\,\mathbf{1}\{S(u) < S^*_T(u)\} \mid S(t)=S] \, du.
\]

Interpretation for our codebase (risk-neutral, continuous dividend yield `q`):
- In our models, drift is typically $\mu = r-q$ between dividend/event dates, so $(r-\mu)=q$.
- Define the two discounted building blocks:
  - **Digital put** price (maturity $u$, strike $B$):
    \( D(t,S;u,B) := e^{-r(u-t)}\,\mathbb{E}[\mathbf{1}\{S(u)<B\}\mid S(t)=S] \).
  - **Asset-or-nothing put** price (maturity $u$, strike $B$):
    \( A(t,S;u,B) := e^{-r(u-t)}\,\mathbb{E}[S(u)\,\mathbf{1}\{S(u)<B\}\mid S(t)=S] \).

Then Eq. 11 becomes an EEP integral in *tradable prices* we can compute via COS:

\[
P(t,S) = p(t,S) + rK \int_t^T D(t,S;u,S^*_T(u))\,du - q\int_t^T A(t,S;u,S^*_T(u))\,du.
\]

### Boundary condition
Boundary is defined by value matching (and smooth pasting in diffusion cases):

\[
P(t, S^*_T(t)) = K - S^*_T(t).
\]

Plugging the EEP representation into this gives a nonlinear integral equation for $S^*_T(t)$.
For GBM + proportional dividends, Andersen derives an explicit form (Eq. 12) and fixed-point form (Eq. 13). For our **Levy/CHF** setting we likely keep the *structure* but compute the needed quantities via COS (digitals / AON) rather than normal-CDF closed forms.

## What we already have (reusable)

- `american_options/engine.py`:
  - `CharacteristicFunction.increment_char(u, dt)`: lets us price conditional on an arbitrary spot without re-instantiating models by multiplying by `exp(i u ln(S))`.
  - `COSPricer.european_price(...)`: already implements Fang–Oosterlee COS for vanilla call/put.
  - A stable truncation-domain selection via model cumulants.

- Levy-style models and characteristic functions are already present (`GBMCHF`, `MertonCHF`, `VGCHF`, `CGMYCHF`, etc.).

## What we need to add

### 1) COS building blocks for EEP
Add COS pricing for the two payoffs required by Eq. 11:

- Digital put payoff: $g(x)=\mathbf{1}\{x < \ln B\}$ (where $x=\ln S_T$).
- Asset-or-nothing put payoff: $g(x)=e^x\mathbf{1}\{x < \ln B\}$.

Plan:
- Implement analytic COS coefficients for both payoffs (Fang–Oosterlee-style), similar to the existing `psi`/`chi` coefficients used for vanilla.
- Expose as new methods on `COSPricer`:
  - `digital_put_price(B, T, spot=None, event=None) -> np.ndarray`
  - `asset_or_nothing_put_price(B, T, spot=None, event=None) -> np.ndarray`

Implementation detail:
- Use `increment_char(u, dt)` and then multiply by `exp(i u ln(spot))` so we can price the same model at many boundary spots efficiently.
- For truncation bounds, reuse `(c1,c2,c4)=model.cumulants(dt)` but shift `c1` by `ln(spot / model.S0)` (higher cumulants typically spot-independent for exponential Levy).

### 2) EEP integration given a boundary curve
Add a function:

- `eep_from_boundary(model, K, T, t_grid, B_grid, spot=None, quad="simpson")`

that computes:
\[
\mathrm{EEP}(t,S) = rK\int_t^T D(t,S;u,B(u))du - q\int_t^T A(t,S;u,B(u))du.
\]

with a chosen quadrature over `u` (simple trapezoid first; Simpson optional).

### 3) Boundary solver
Provide a boundary solver that returns a discrete curve `B(t_i)`.

Minimum viable (works for GBM / Levy without discrete dividends):
- Choose a time grid `t_i` (e.g., uniform in $[0,T]$).
- Initialize `B_i` with a simple guess (e.g., `B_i = K` or a monotone curve ending at Andersen’s terminal condition).
- Iterate until convergence:
  - For each `t_i` (backward in time), solve the scalar equation
    \( K - B_i - P(t_i, B_i; B_{i:}) = 0 \)
    using bisection (robust) where `P` is computed from European + EEP integrals.

Notes:
- This is naturally **O(M^2)** in the number of time nodes `M` if done naively; start there for correctness.
- Performance improvements later:
  - Cache `increment_char(u, dt_j)` for each unique `dt`.
  - Vectorize over strikes `B(u)` at fixed `dt` where possible.

### 4) New pricer entry point (keep both approaches)
Add a second American pricing path without disturbing the existing rollback method.

Proposed API:
- `CharacteristicFunction.american_price_eep(K, T, *, boundary="solve", steps=200, N=1024, L=10, ...)`
  - returns `(american, european, eep, boundary_curve)` optionally.

Keep existing:
- `CharacteristicFunction.american_price(...)` (COS rollback) unchanged.

## Validation strategy

1. **Digital/AON sanity** under GBM:
   - Compare `digital_put_price` vs Black–Scholes digital closed form.
   - Compare `asset_or_nothing_put_price` vs known analytic formula.

2. **EEP representation** with an external boundary:
   - Use a boundary curve inferred from the existing rollback (approximate) and check that `European + EEP` matches the rollback American price.

3. **Boundary solve correctness**:
   - For GBM/no dividends, compare EEP-method American to rollback American across strikes.

4. **Levy models**:
   - Run the same regression comparisons for Merton/VG/CGMY.

## Known scope limitations (initial cut)

- Discrete cash/proportional dividends and event jumps complicate the boundary (paper has results for proportional dividends at fixed dates). We should first deliver a clean continuous-yield implementation; then extend to discrete events by splitting the integrals across event intervals and applying the event transform to the CHF.

## Where to look in the workspace

- Extracted paper text around Eq. 11: `tools/andersen_eq11_page.txt` and `tools/andersen_eq11_snippet.txt`.
- Existing COS payoff coefficient patterns: `american_options/engine.py` inside `COSPricer.european_price`.
