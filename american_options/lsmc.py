import numpy as np


class LSMCPricer:
    """Longstaff–Schwartz Monte Carlo pricer for American calls under GBM with proportional discrete dividends.

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
        self.divs = divs
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

    def _train_regressors(self, K, tau, steps, n_train, deg):
        # simulate training paths
        divs_shifted = {t: v for t, v in self._shift_divs(0.0, tau).items()}  # already relative to start
        S = self._simulate_paths(self.S0, tau, n_train, steps, divs_shifted)
        pay = np.maximum(S - K, 0.0)
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

    def _price_with_coeffs(self, K, tau, steps, coeffs_per_step, n_price, deg):
        divs_shifted = {t: v for t, v in self._shift_divs(0.0, tau).items()}
        S = self._simulate_paths(self.S0, tau, n_price, steps, divs_shifted)
        pay = np.maximum(S - K, 0.0)
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

    def _refit_on_price(self, K, tau, steps, n_price, deg):
        # simulate pricing paths and fit regressors on them (second sweep), then price
        divs_shifted = {t: v for t, v in self._shift_divs(0.0, tau).items()}
        S = self._simulate_paths(self.S0, tau, n_price, steps, divs_shifted)
        pay = np.maximum(S - K, 0.0)
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

    def price_at_tau(self, K, tau, steps=40, n_train=2000, n_price=2000, deg=3):
        """Return (train_price, train_cont0, refit_price, refit_cont0) for a given remaining horizon tau."""
        coeffs = self._train_regressors(K, tau, steps, n_train, deg)
        train_price, train_cont0 = self._price_with_coeffs(K, tau, steps, coeffs, n_price, deg)
        refit_price, refit_cont0 = self._refit_on_price(K, tau, steps, n_price, deg)
        return train_price, train_cont0, refit_price, refit_cont0
