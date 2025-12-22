import numpy as np
from american_options.engine import COSPricer, GBMCHF

def test_american_gbm():
    S0 = 100.0
    K = 105.0
    r = 0.03
    q = 0.05 # Continuous dividend
    T = 1.0
    vol = 0.2
    
    params = {"vol": vol}
    model = GBMCHF(S0=S0, r=r, q=q, divs={}, params=params)
    
    pricer = COSPricer(model, N=512, L=10, M=1024)
    
    # For q=0, American Call == European Call
    euro_price = pricer.european_price(np.array([K]), T)[0]
    am_price = pricer.american_price(np.array([K]), T, steps=10)[0]
    
    print(f"GBM (q=0): Euro={euro_price:.6f}, American={am_price:.6f}")
    print(f"Difference: {am_price - euro_price:.6f}")

if __name__ == "__main__":
    test_american_gbm()
