import numpy as np
from american_options import CGMYCHF
from american_options.engine import COSPricer
from scipy.stats import norm
from scipy.special import gamma

def bs_call(S, K, r, q, vol, T):
    sigma = vol
    if sigma * np.sqrt(T) <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def test_convergence():
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    vol = 0.25
    K = 100.0
    bs_ref = bs_call(S0, K, r, q, vol, T)
    
    print(f"{'M=G':>10} | {'C':>15} | {'COS Price':>15} | {'Diff':>12}")
    print("-" * 60)
    
    Y = 0.5
    # Test powers of 10
    for p in range(1, 16):
        M = 10.0**p
        G = M
        # C = sigma^2 / (2 * Gamma(2-Y) * M^(Y-2))
        C = (vol**2) / (2 * gamma(2.0 - Y) * (M**(Y - 2.0)))
        
        # Stable psi calculation for testing
        def stable_psi(z, C, G, M, Y):
            gamma_m = gamma(-Y)
            # (M - 1j*z)^Y - M^Y = M^Y * ((1 - 1j*z/M)^Y - 1)
            # Use expm1(Y * log1p(-1j*z/M))
            term1 = M**Y * np.expm1(Y * np.log1p(-1j * z / M))
            term2 = G**Y * np.expm1(Y * np.log1p(1j * z / G))
            return C * gamma_m * (term1 + term2)

        try:
            # We can't easily inject stable_psi into the class without monkeypatching
            # but we can check if the stable_psi itself is stable
            z = 1.0
            val_stable = stable_psi(z, C, G, M, Y)
            
            # Compare with BS exponent: -0.5 * vol^2 * z^2
            bs_exponent = -0.5 * vol**2 * z**2
            # Note: CGMY psi includes a drift term that BS doesn't have in its 'psi'
            # The real part should match.
            # print(f"10^{p:<2} | Real(psi): {val_stable.real:12.6f} | BS: {-bs_exponent:12.6f}")
            
            model = CGMYCHF(S0, r, q, {}, {"C": C, "G": G, "M": M, "Y": Y})
            pricer = COSPricer(model, N=2**12, L=10.0)
            price = pricer.european_price(np.array([K]), T)[0]
            diff = price - bs_ref
            # print(f"10^{p:<2}      | {C:15.2e} | {price:15.10f} | {diff:12.3e}")
            print(f"10^{p:<2}  BS {bs_ref:15.10f} CGMY {price:15.10f} | {diff:12.3e}")
        except Exception as e:
            print(f"10^{p:<2}      | FAILED: {e}")

if __name__ == "__main__":
    test_convergence()
