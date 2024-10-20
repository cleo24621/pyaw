# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/18
@DESCRIPTION: 
"""
import matplotlib.pyplot as plt
import numpy as np

va = 1295  # km/s
te = 42  # eV
rho_i = 8.4  # km
rho_s = 2.3  # km
lambda_e = 1.0  # km
f = np.linspace(1e-3,1e1,100)
v_fit = 81.8

def get_ratio(va,f,rho_i,rho_s,lambda_e):
    k_transverse = (2 * np.pi * f) / v_fit
    return va * np.sqrt((1 + (k_transverse ** 2) * (lambda_e ** 2)) / (1 + (k_transverse ** 2) * (rho_i ** 2 + rho_s ** 2))) * (1 + (k_transverse ** 2) * (rho_i ** 2))

plt.figure()
plt.plot(f,get_ratio(va,f,rho_i,rho_s,lambda_e) * 1e3)
plt.xscale('log')
plt.yscale('log')
plt.yticks([1e5,1e6,1e7,1e8,1e9],['1e5','1e6','1e7','1e8','1e9'])
plt.show()

#%%
# e b fre
# psd -> e/b
# wavelet -> coherency and relative phase
# rise from va
# in the cusp,