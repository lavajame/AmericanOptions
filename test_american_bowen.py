import numpy as np
from american_options.engine import COSPricer, CGMYCHF

def test_american_bowen():
    # Parameters from Bowen.pdf Table 4
    # Y=1.5, C=1, G=5, M=5, q=0.05, r=0.1, S0=100, K=110, T=1.0
    S0 = 100.0
    K = 110.0
    r = 0.1
    q = 0.05
    T = 1.0
    
    # CGMY parameters
    C = 1.0
    G = 5.0
    M_cgmy = 5.0
    Y = 1.5
    
    params = {"C": C, "G": G, "M": M_cgmy, "Y": Y}
    model = CGMYCHF(S0=S0, r=r, q=q, divs={}, params=params)
    
    # COS parameters
    # Bowen uses N=512 or 1024. Let's use N=512.
    # L=10 is usually enough.
    pricer = COSPricer(model, N=512, L=10, M=1024) # M is grid size in current implementation
    
    # Bowen uses Richardson extrapolation on Bermudan prices.
    # Let's compute Bermudan prices for steps = 8, 16, 32, 64
    steps_list = [8, 16, 32, 64]
    v_bermudan = []
    
    print(f"Pricing American Call under CGMY (Y={Y}, q={q})")
    print(f"S0={S0}, K={K}, r={r}, T={T}")
    
    # Check European price
    euro_price = pricer.european_price(np.array([K]), T)[0]
    print(f"European Price: {euro_price:.6f}")
    
    for steps in steps_list:
        # The current american_price method returns the price at S0
        price = pricer.american_price(np.array([K]), T, steps=steps, rollback_debug=True)
        v_bermudan.append(price[0])
        print(f"Steps={steps:2d}, Price={price[0]:.6f}")
        
    # Richardson extrapolation (4-point)
    # v_am = (1/21) * (64*v(8M) - 56*v(4M) + 14*v(2M) - v(M))
    # Here M=8, so 8M=64, 4M=32, 2M=16, M=8
    v8, v16, v32, v64 = v_bermudan
    v_am = (1.0/21.0) * (64*v64 - 56*v32 + 14*v16 - v8)
    
    print(f"\nRichardson Extrapolated American Price: {v_am:.6f}")
    print(f"Bowen Reference Value (Table 4): 44.0934")
    print(f"Difference: {v_am - 44.0934:.6f}")

if __name__ == "__main__":
    test_american_bowen()
