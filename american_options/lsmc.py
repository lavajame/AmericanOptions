import numpy as np

from american_options.engine import cash_divs_to_proportional_divs


def _payoff(S: np.ndarray, K: float, *, is_call: bool) -> np.ndarray:
    if is_call:
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


class LSMCPricer:
    """Longstaff–Schwartz Monte Carlo pricer for American options under GBM with discrete dividends.

    Project-wide convention: dividends are specified as cash amounts in spot currency:
        divs[t] = (D_mean, D_std)
    This pricer converts them internally to the proportional form used by the simulation.

    Usage:
        pr = LSMCPricer(S0, r, q, divs, vol, seed=42)
        train_price, refit_price = pr.price_at_tau(K, tau, n_train=2000, n_price=2000, steps=40, deg=3)

    The pricer returns two prices:
    - train_price: price computed using regressors fitted on an independent training set.
    - refit_price: price computed by refitting regression on the pricing paths (second sweep).
    Both are Monte Carlo estimates and have sampling noise.
    """

    def __init__(self, S0, r, q, divs, vol, seed: int = 2025):
        self.S0 = S0
        self.r = r
        self.q = q
        # Convert once to the internal proportional form expected by the simulation code.
        self.divs = cash_divs_to_proportional_divs(self.S0, self.r, self.q, divs)
        self.vol = vol
        self.rng = np.random.default_rng(seed)

    def _shift_divs(self, t_current, T):
        # Return dictionary of dividends relative to remaining horizon tau = T - t_current
        return {t_div - t_current: (m, s) for t_div, (m, s) in self.divs.items() if t_div > t_current + 1e-12}

    def _simulate_paths(self, S0, tau, n_paths, steps, divs_shifted):
        dt = tau / steps if steps > 0 else tau
        S = np.empty((n_paths, steps + 1), dtype=float)
        S[:, 0] = S0
        vol = self.vol
        drift = (self.r - self.q - 0.5 * vol * vol)
        for k in range(1, steps + 1):
            z = self.rng.standard_normal(n_paths)
            S[:, k] = S[:, k - 1] * np.exp(drift * dt + vol * np.sqrt(dt) * z)
            # apply dividends that fall exactly at this step
            t_k = k * dt
            # check for dividends at this t_k (allow small tolerance)
            for t_div, (m, _) in divs_shifted.items():
                if abs(t_div - t_k) <= 0.5 * dt:
                    S[:, k] *= (1.0 - m)
        return S

    def _basis(self, S, deg):
        # polynomial basis in S/S0: [1, x, x^2, ..., x^deg]
        x = S / self.S0
        # shape (n_samples, deg+1)
        cols = [np.ones_like(x)]
        for d in range(1, deg + 1):
            cols.append(x ** d)
        return np.vstack(cols).T

    def _train_regressors(self, K, tau, steps, n_train, deg, *, is_call: bool):
        # simulate training paths
        divs_shifted = {t: v for t, v in self._shift_divs(0.0, tau).items()}  # already relative to start
        S = self._simulate_paths(self.S0, tau, n_train, steps, divs_shifted)
        pay = _payoff(S, K, is_call=is_call)
        dt = tau / steps if steps > 0 else tau
        df = np.exp(-self.r * dt)

        # Initialize cashflows at maturity
        cash = pay[:, -1].copy()
        coeffs_per_step = [None] * steps

        # Backward induction for regressors
        for j in range(steps - 1, -1, -1):
            St = S[:, j]
            immediate = pay[:, j]
            # discount next cash to current time
            discounted = cash * df
            itm = immediate > 0
            if np.any(itm):
                X = self._basis(St[itm], deg)
                Y = discounted[itm]
                # linear regression with small ridge
                lam = 1e-8
                A = X.T @ X + lam * np.eye(X.shape[1])
                b = X.T @ Y
                coeffs = np.linalg.solve(A, b)
                coeffs_per_step[j] = coeffs
                cont_pred = X @ coeffs
                exercise = itm.copy()
                exercise[itm] = immediate[itm] > cont_pred
                # update cashflows
                cash[exercise] = immediate[exercise]
                cash[~exercise] = discounted[~exercise]
            else:
                # no ITM paths: simply discount
                cash = discounted
                coeffs_per_step[j] = None
        # At end, cash contains value at time 0 on training sample
        return coeffs_per_step

    def _price_with_coeffs(self, K, tau, steps, coeffs_per_step, n_price, deg, *, is_call: bool):
        divs_shifted = {t: v for t, v in self._shift_divs(0.0, tau).items()}
        S = self._simulate_paths(self.S0, tau, n_price, steps, divs_shifted)
        pay = _payoff(S, K, is_call=is_call)
        dt = tau / steps if steps > 0 else tau
        df = np.exp(-self.r * dt)

        cash = pay[:, -1].copy()
        # record continuation predictions at time 0 for S0
        cont0_pred = None
        for j in range(steps - 1, -1, -1):
            St = S[:, j]
            immediate = pay[:, j]
            discounted = cash * df
            coeffs = coeffs_per_step[j]
            if coeffs is None:
                cash = discounted
                continue
            X_all = self._basis(St, deg)
            cont_pred_all = X_all @ coeffs
            # exercise when immediate > cont
            exercise = immediate > cont_pred_all
            cash[exercise] = immediate[exercise]
            cash[~exercise] = discounted[~exercise]
            # predicted continuation at S0 for this j
            if j == 0:
                phi0 = self._basis(np.array([self.S0]), deg)[0]
                cont0_pred = float(phi0 @ coeffs)
        price = float(np.mean(cash))
        return price, cont0_pred

    def _refit_on_price(self, K, tau, steps, n_price, deg, *, is_call: bool):
        # simulate pricing paths and fit regressors on them (second sweep), then price
        divs_shifted = {t: v for t, v in self._shift_divs(0.0, tau).items()}
        S = self._simulate_paths(self.S0, tau, n_price, steps, divs_shifted)
        pay = _payoff(S, K, is_call=is_call)
        dt = tau / steps if steps > 0 else tau
        df = np.exp(-self.r * dt)

        cash = pay[:, -1].copy()
        coeffs_per_step = [None] * steps
        for j in range(steps - 1, -1, -1):
            St = S[:, j]
            immediate = pay[:, j]
            discounted = cash * df
            itm = immediate > 0
            if np.any(itm):
                X = self._basis(St[itm], deg)
                Y = discounted[itm]
                lam = 1e-8
                A = X.T @ X + lam * np.eye(X.shape[1])
                b = X.T @ Y
                coeffs = np.linalg.solve(A, b)
                coeffs_per_step[j] = coeffs
                cont_pred = X @ coeffs
                exercise = itm.copy()
                exercise[itm] = immediate[itm] > cont_pred
                cash[exercise] = immediate[exercise]
                cash[~exercise] = discounted[~exercise]
            else:
                cash = discounted
                coeffs_per_step[j] = None
        # Now price using these coeffs on the same paths but treat as pricing pass
        # we want price at S0: evaluate exercise using coeffs_per_step
        # perform a new pricing pass using same S
        cash2 = pay[:, -1].copy()
        cont0_pred = None
        for j in range(steps - 1, -1, -1):
            St = S[:, j]
            immediate = pay[:, j]
            discounted = cash2 * df
            coeffs = coeffs_per_step[j]
            if coeffs is None:
                cash2 = discounted
                continue
            X_all = self._basis(St, deg)
            cont_pred_all = X_all @ coeffs
            exercise = immediate > cont_pred_all
            cash2[exercise] = immediate[exercise]
            cash2[~exercise] = discounted[~exercise]
            if j == 0 and coeffs is not None:
                phi0 = self._basis(np.array([self.S0]), deg)[0]
                cont0_pred = float(phi0 @ coeffs)
        price = float(np.mean(cash2))
        return price, cont0_pred

    def price_at_tau(self, K, tau, steps=40, n_train=2000, n_price=2000, deg=3, *, is_call: bool = True):
        """Return (train_price, train_cont0, refit_price, refit_cont0) for a given remaining horizon tau."""
        coeffs = self._train_regressors(K, tau, steps, n_train, deg, is_call=is_call)
        train_price, train_cont0 = self._price_with_coeffs(K, tau, steps, coeffs, n_price, deg, is_call=is_call)
        refit_price, refit_cont0 = self._refit_on_price(K, tau, steps, n_price, deg, is_call=is_call)
        return train_price, train_cont0, refit_price, refit_cont0


class LSMCCompositePricer:
    """LSMC pricer for American options under CompositeLevyCHF-like dynamics.

    Supports component simulation for:
    - gbm (diffusion only)
    - merton (diffusion + Gaussian jumps)
    - kou (diffusion + double-exponential jumps)
    - vg (Variance Gamma)
    - nig (Normal-Inverse-Gaussian)

    Not supported:
    - cgmy (no MC sampler implemented in this repo)

    Dividends:
    - input `divs` are cash dividends (mean, std). Mean is applied deterministically.
    """

    def __init__(self, *, S0: float, r: float, q: float, divs: dict[float, tuple[float, float]], model_params: dict, seed: int = 2025):
        # Store cash-dividend schedule converted to proportional.
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.divs = cash_divs_to_proportional_divs(self.S0, self.r, self.q, divs)
        self.model_params = dict(model_params)
        self.rng = np.random.default_rng(int(seed))

    @staticmethod
    def _psi_minus_i(components: list[dict]) -> complex:
        # Minimal psi(-i) from components; used for martingale correction.
        u_mi = -1j
        total = 0.0 + 0.0j
        for comp in components:
            ctype = str(comp["type"]).lower().strip()
            p = comp.get("params", {})
            if ctype == "gbm":
                sigma = float(p.get("sigma", p.get("vol", 0.0)))
                total += -0.5 * (u_mi ** 2) * (sigma ** 2)
            elif ctype == "merton":
                sigma = float(p.get("sigma", p.get("vol", 0.0)))
                lam = float(p.get("lam", 0.0))
                muJ = float(p.get("muJ", 0.0))
                sigmaJ = float(p.get("sigmaJ", 0.0))
                phi_jump = np.exp(1j * u_mi * muJ - 0.5 * (u_mi ** 2) * (sigmaJ ** 2))
                total += -0.5 * (u_mi ** 2) * (sigma ** 2) + lam * (phi_jump - 1.0)
            elif ctype == "kou":
                sigma = float(p.get("sigma", p.get("vol", 0.0)))
                lam = float(p.get("lam", 0.0))
                pp = float(p.get("p", 0.5))
                eta1 = float(p.get("eta1", 10.0))
                eta2 = float(p.get("eta2", 5.0))
                phi_jump = pp * (eta1 / (eta1 - 1j * u_mi)) + (1.0 - pp) * (eta2 / (eta2 + 1j * u_mi))
                total += -0.5 * (u_mi ** 2) * (sigma ** 2) + lam * (phi_jump - 1.0)
            elif ctype == "vg":
                theta = float(p.get("theta", 0.0))
                sigma = float(p.get("sigma", 0.0))
                nu = float(p.get("nu", 0.0))
                inside = 1.0 - 1j * theta * nu * u_mi + 0.5 * (sigma ** 2) * nu * (u_mi ** 2)
                total += -(1.0 / nu) * np.log(inside)
            elif ctype == "nig":
                alpha = float(p.get("alpha"))
                beta = float(p.get("beta"))
                delta = float(p.get("delta"))
                mu = float(p.get("mu", 0.0))
                gamma = np.sqrt((alpha * alpha) - (beta * beta) + 0j)
                sqrt_term = np.sqrt((alpha * alpha) - (beta + 1j * u_mi) ** 2 + 0j)
                total += 1j * u_mi * mu + delta * (gamma - sqrt_term)
            elif ctype == "cgmy":
                raise NotImplementedError("CGMY MC simulation not implemented")
            else:
                raise ValueError(f"Unsupported component type for LSMC simulation: {ctype}")
        return complex(total)

    def _shift_divs(self, t_current: float, T: float) -> dict[float, tuple[float, float]]:
        return {t_div - t_current: (m, s) for t_div, (m, s) in self.divs.items() if t_div > t_current + 1e-12}

    @staticmethod
    def _basis(S: np.ndarray, S0: float, deg: int) -> np.ndarray:
        x = S / float(S0)
        cols = [np.ones_like(x)]
        for d in range(1, int(deg) + 1):
            cols.append(x ** d)
        return np.vstack(cols).T

    def _simulate_paths(self, *, tau: float, n_paths: int, steps: int, components: list[dict]) -> np.ndarray:
        tau = float(tau)
        steps = int(steps)
        dt = tau / steps if steps > 0 else tau
        S = np.empty((n_paths, steps + 1), dtype=float)
        logS = np.empty((n_paths, steps + 1), dtype=float)
        S[:, 0] = self.S0
        logS[:, 0] = np.log(self.S0)

        psi_mi = self._psi_minus_i(components)
        drift = (self.r - self.q) - float(np.real_if_close(psi_mi, tol=1e6))

        # dividends (relative to start)
        divs_shifted = self._shift_divs(0.0, tau)

        for k in range(1, steps + 1):
            incr = np.zeros(n_paths, dtype=float)

            for comp in components:
                ctype = str(comp["type"]).lower().strip()
                p = comp.get("params", {})

                if ctype == "gbm":
                    sigma = float(p.get("sigma", p.get("vol", 0.0)))
                    z = self.rng.standard_normal(n_paths)
                    incr += sigma * np.sqrt(dt) * z

                elif ctype == "merton":
                    sigma = float(p.get("sigma", p.get("vol", 0.0)))
                    lam = float(p.get("lam", 0.0))
                    muJ = float(p.get("muJ", 0.0))
                    sigmaJ = float(p.get("sigmaJ", 0.0))
                    z = self.rng.standard_normal(n_paths)
                    incr += sigma * np.sqrt(dt) * z
                    nJ = self.rng.poisson(lam * dt, size=n_paths)
                    if np.any(nJ > 0):
                        jump_sum = np.zeros(n_paths, dtype=float)
                        idx = np.where(nJ > 0)[0]
                        for i in idx:
                            jump_sum[i] = float(np.sum(self.rng.normal(loc=muJ, scale=sigmaJ, size=int(nJ[i]))))
                        incr += jump_sum

                elif ctype == "kou":
                    sigma = float(p.get("sigma", p.get("vol", 0.0)))
                    lam = float(p.get("lam", 0.0))
                    pp = float(p.get("p", 0.5))
                    eta1 = float(p.get("eta1", 10.0))
                    eta2 = float(p.get("eta2", 5.0))
                    z = self.rng.standard_normal(n_paths)
                    incr += sigma * np.sqrt(dt) * z
                    nJ = self.rng.poisson(lam * dt, size=n_paths)
                    if np.any(nJ > 0):
                        jump_sum = np.zeros(n_paths, dtype=float)
                        idx = np.where(nJ > 0)[0]
                        for i in idx:
                            n = int(nJ[i])
                            signs = self.rng.uniform(size=n) < pp
                            mags_pos = self.rng.exponential(scale=1.0 / eta1, size=int(np.sum(signs)))
                            mags_neg = self.rng.exponential(scale=1.0 / eta2, size=int(np.sum(~signs)))
                            jump_sum[i] = float(np.sum(mags_pos) - np.sum(mags_neg))
                        incr += jump_sum

                elif ctype == "vg":
                    theta = float(p.get("theta", 0.0))
                    sigma = float(p.get("sigma", 0.0))
                    nu = float(p.get("nu", 0.0))
                    if nu <= 0.0:
                        raise ValueError("VG requires nu > 0")
                    # G ~ Gamma(shape=dt/nu, scale=nu)
                    G = self.rng.gamma(shape=dt / nu, scale=nu, size=n_paths)
                    z = self.rng.standard_normal(n_paths)
                    incr += theta * G + sigma * np.sqrt(G) * z

                elif ctype == "nig":
                    alpha = float(p.get("alpha"))
                    beta = float(p.get("beta"))
                    delta = float(p.get("delta"))
                    mu = float(p.get("mu", 0.0))
                    if alpha <= 0.0 or delta <= 0.0 or alpha <= abs(beta):
                        raise ValueError("Invalid NIG params")
                    gamma = float(np.sqrt((alpha * alpha) - (beta * beta)))
                    # W ~ IG(mean=delta*dt/gamma, scale=(delta*dt)^2)
                    mean = (delta * dt) / gamma
                    scale = (delta * dt) ** 2
                    W = self.rng.wald(mean=mean, scale=scale, size=n_paths)
                    z = self.rng.standard_normal(n_paths)
                    incr += (mu * dt) + beta * W + np.sqrt(W) * z

                else:
                    raise ValueError(f"Unsupported component type: {ctype}")

            logS[:, k] = logS[:, k - 1] + drift * dt + incr
            S[:, k] = np.exp(logS[:, k])

            # apply proportional cash dividend means at this step time
            t_k = k * dt
            for t_div, (m, _) in divs_shifted.items():
                if abs(t_div - t_k) <= 0.5 * dt:
                    S[:, k] *= (1.0 - float(m))
                    logS[:, k] = np.log(S[:, k])

        return S

    def price_at_tau(
        self,
        *,
        K: float,
        tau: float,
        components: list[dict],
        steps: int = 40,
        n_train: int = 20_000,
        n_price: int = 20_000,
        deg: int = 3,
        is_call: bool = False,
    ) -> tuple[float, float, float, float]:
        # Train regressors on independent training sample
        S_train = self._simulate_paths(tau=tau, n_paths=n_train, steps=steps, components=components)
        pay_train = _payoff(S_train, float(K), is_call=is_call)
        dt = float(tau) / int(steps) if steps > 0 else float(tau)
        df = np.exp(-self.r * dt)

        cash = pay_train[:, -1].copy()
        coeffs_per_step: list[np.ndarray | None] = [None] * int(steps)
        for j in range(int(steps) - 1, -1, -1):
            St = S_train[:, j]
            immediate = pay_train[:, j]
            discounted = cash * df
            itm = immediate > 0
            if np.any(itm):
                X = self._basis(St[itm], self.S0, int(deg))
                Y = discounted[itm]
                lam = 1e-8
                A = X.T @ X + lam * np.eye(X.shape[1])
                b = X.T @ Y
                coeffs = np.linalg.solve(A, b)
                coeffs_per_step[j] = coeffs
                cont_pred = X @ coeffs
                exercise = itm.copy()
                exercise[itm] = immediate[itm] > cont_pred
                cash[exercise] = immediate[exercise]
                cash[~exercise] = discounted[~exercise]
            else:
                cash = discounted

        # Price pass using frozen coeffs
        S_price = self._simulate_paths(tau=tau, n_paths=n_price, steps=steps, components=components)
        pay_price = _payoff(S_price, float(K), is_call=is_call)
        cash2 = pay_price[:, -1].copy()
        cont0_pred = np.nan
        for j in range(int(steps) - 1, -1, -1):
            St = S_price[:, j]
            immediate = pay_price[:, j]
            discounted = cash2 * df
            coeffs = coeffs_per_step[j]
            if coeffs is None:
                cash2 = discounted
                continue
            X_all = self._basis(St, self.S0, int(deg))
            cont_pred_all = X_all @ coeffs
            exercise = immediate > cont_pred_all
            cash2[exercise] = immediate[exercise]
            cash2[~exercise] = discounted[~exercise]
            if j == 0:
                phi0 = self._basis(np.array([self.S0]), self.S0, int(deg))[0]
                cont0_pred = float(phi0 @ coeffs)

        train_price = float(np.mean(cash))
        train_cont0 = float("nan")
        refit_price = float(np.mean(cash2))
        refit_cont0 = float(cont0_pred)
        return train_price, train_cont0, refit_price, refit_cont0
