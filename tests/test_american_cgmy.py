import numpy as np
from american_options import CGMYCHF
from american_options.engine import COSPricer

# Parameters from COS.pdf for American CGMY Y=1.5
S0 = 100.0
r = 0.1
q = 0.05
divs = {}
T = 1.0
K = np.array([100.0])
params = {"C": 1.0, "G": 5.0, "M": 5.0, "Y": 1.5}

model = CGMYCHF(S0, r, q, divs, params)
pricer = COSPricer(model, N=512, L=10.0)

# European
euro = pricer.european_price(K, T)
print('European price:', euro[0])

# American
amer = pricer.american_price(K, T, steps=50)
print('American price:', amer[0])

# Reference from PDF: 44.0934