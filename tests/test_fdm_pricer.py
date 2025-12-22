import numpy as np
from plot_diagnostics import FDMPricer


def shift_divs_for_time(divs, t_current):
    return {dt - t_current: (m, s) for dt, (m, s) in divs.items() if dt > t_current + 1e-12}


def test_fdm_nonnegative_and_american_ge_european():
    S0, r, q, vol = 100.0, 0.02, 0.0, 0.25
    K = 100.0
    divs = {0.25: (0.02, 1e-10), 0.75: (0.02, 1e-10)}
    times = [0.0, 0.24, 0.249, 0.251, 0.5, 0.74, 0.749, 0.751]

    for t in times:
        tau = 1.0 - t
        divs_shift = shift_divs_for_time(divs, t)
        fdm = FDMPricer(S0, r, q, vol, divs_shift, NS=200, NT=max(100, int(300 * tau)))
        e = fdm.price(K, tau, american=False)
        a = fdm.price(K, tau, american=True)
        assert e >= -1e-8, f"FDM European negative at t={t}: {e}"
        assert a + 1e-8 >= e, f"FDM American < European at t={t}: a={a}, e={e}"
