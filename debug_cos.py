import numpy as np
from american_options import GBMCHF
from american_options.engine import COSPricer, _dividend_adjustment

S0=100; r=0.02; q=0.01; divs={0.5:(0.03,0.01)}; params={'vol':0.25}
model=GBMCHF(S0,r,q,divs,params)
pr= COSPricer(model,N=8,L=8.0)
K=np.array([90.,100.,110.]);T=1.0
exp_divs, div_params = _dividend_adjustment(T, model.divs)
var_div = float(np.sum(div_params[:,1])) if div_params.size else 0.0
var = model._var2(T) + var_div
F = forward_price = (model.S0 * np.exp((model.r-model.q)*T - exp_divs))
mu = np.log(F) - 0.5 * var
print('var',var,'mu',mu)
print('a,b', mu - pr.L*np.sqrt(max(var,1e-16)), mu + pr.L*np.sqrt(max(var,1e-16)))
prices = pr.european_price(K,T)
print('european prices', prices)
# inspect terms
N=pr.N
k = np.arange(N).reshape((-1,1))
exp_divs, div_params = _dividend_adjustment(T, model.divs)
var_div = float(np.sum(div_params[:,1])) if div_params.size else 0.0
var = model._var2(T) + var_div
F = forward_price = (model.S0 * np.exp((model.r-model.q)*T - exp_divs))
mu = np.log(F) - 0.5 * var
b = mu + pr.L*np.sqrt(max(var,1e-16))
a = mu - pr.L*np.sqrt(max(var,1e-16))
print('a,b',a,b)

u = (k * np.pi / (b - a)).flatten()
phi = model.char_func(u,T)
print('phi[0]',phi[0])
# compute Vk for K=100 (c)
c = np.log(np.array([100.])).reshape((1,-1))
kpi = k * np.pi / (b - a)
cos_term_c = np.cos(kpi * (c - a))
cos_term_d = np.cos(kpi * (b - a))
sin_term_c = np.sin(kpi * (c - a))
sin_term_d = np.sin(kpi * (b - a))
denom = 1.0 + (kpi ** 2)
chi = (np.exp(b) * (cos_term_d + kpi * sin_term_d) - np.exp(c) * (cos_term_c + kpi * sin_term_c)) / denom
kpi_b = kpi
numerator = (sin_term_d - sin_term_c)
safe_div = np.zeros_like(numerator)
np.divide(numerator, kpi_b, out=safe_div, where=~np.isclose(kpi_b, 0.0))
psi = np.where(np.isclose(kpi_b, 0.0), b - c, safe_div)
Vk = 2.0 / (b - a) * (chi - np.exp(c) * psi)
print('Vk[0]',Vk[0,0])
print('sum_k phi*exp(-i*u*a)*Vk', np.real(np.sum(phi.reshape((-1,1))*np.exp(-1j*(k*np.pi/(b-a))*a)*Vk,axis=0))[0])
print('discounted',np.exp(-model.r*T)*np.real(np.sum(phi.reshape((-1,1))*np.exp(-1j*(k*np.pi/(b-a))*a)*Vk,axis=0))[0])
