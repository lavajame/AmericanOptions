import os
import sys
import numpy as np
# ensure workspace root is on sys.path so local package imports work when running as a script
sys.path.insert(0, os.getcwd())
from american_options import CGMYCHF
from american_options.engine import COSPricer

os.makedirs('figs', exist_ok=True)

S0 = 100.0
r = 0.1
q = 0.05
K = 110.0
Y = 1.98
params = {"C": 1.0, "G": 5.0, "M": 5.0, "Y": Y}
model = CGMYCHF(S0, r, q, {}, params)
pr = COSPricer(model, N=1024, L=8.0)
Karr = np.array([K])

# cleanup diagnostics file
diag_file = 'figs/rollback_diag.csv'
try:
    os.remove(diag_file)
except Exception:
    pass

print('Running rollback diagnostic for CGMY Y=1.98...')
model.params['safety_trunc_var'] = 4.0
model.params['debug'] = True
price = pr.american_price(Karr, 1.0, steps=8, use_softmax=False, rollback_debug=True, diagnostics_file=diag_file)[0]
print('Returned price:', price)
print('Diagnostics written to', diag_file)
