import numpy as np
import pandas as pd
from numpy import ndarray

from pyaw.parameters import PhysicalParameters


def get_va(B0: float | ndarray | pd.Series, np: float | ndarray | pd.Series) -> float | ndarray | pd.Series:
    """
    get local alfven velocity
    :param B0: (T SI)
    :param np: (m^{-3}) number density of proton
    :return: (m/s)
    """
    # todo:: may need modify because "n" quality problem.
    return B0 / np.sqrt(PhysicalParameters.mu0 * (PhysicalParameters.mp * np))


def get_ion_gyrofrequency(B: float | ndarray | pd.Series, mi: float, qi=PhysicalParameters.e):
    """
    $\Omega_i$
    :param B: (T SI) measured magnetic field.
    :param qi: (C SI) in the study (in earth ionosphere and magnetosphere), all types of ions charge is e (elementary charge).
    :param mi: (kg SI) ion absolute mass. (consider different ions) may ues weight.
    :return: (rad/s SI) ion gyrofrequency
    """
    return (qi * B) / mi


def get_ion_gyroradius(Ti, mi, Omega_i):
    """
    $\rho_i$
    :param Ti: (K SI)
    :param mi: (kg SI) weight?
    :param Omega_i: (rad/s SI) ion gyrofrequency
    :return: (km SI)
    """
    return np.sqrt(Ti * mi) / Omega_i


def get_ion_acoustic_gyroradius(Te, mi, Omega_i):
    """
    $\rho_s$
    :param Te:
    :param mi:
    :param Omega_i:
    :return:
    """
    return np.sqrt(Te * mi) / Omega_i


def get_electron_inertial_length(ne, me=PhysicalParameters.me, mu0=PhysicalParameters.mu0, e=PhysicalParameters.e):
    """
    $\lambda_e$
    :param me: (kg)
    :param mu0: (H/m)
    :param ne: (cm^{-3} SI)
    :param e: (C)
    :return: (km)
    """
    return np.sqrt(me / (mu0 * ne * e ** 2))

gamma = 3/5  # 绝热指数（通常取 5/3）
k_b = PhysicalParameters.kB
mH = PhysicalParameters.mH

def get_c_s(T_i=2000,m_i=mH):
    """离子声速 $c_s$"""
    return np.sqrt((gamma * k_b * T_i)/(m_i))


def get_Omega_i(B,e=PhysicalParameters.e,m_i=mH):
    """

    Args:
        B: 背景磁场
        e:
        m_i:

    Returns:

    """
    return e * B / m_i

def get_rho_s(c_s,Omega_i):
    """
    离子声回转半径 $rho_s$
    Returns:

    """
    return c_s / Omega_i


def get_beta(n: float, T: pd.Series | np.ndarray, B: pd.Series | np.ndarray):
    """
    $\beta = \frac{p}{p_mag} = \frac{nk_B T}{B^2 / 2\mu_0}$  refer to https://en.wikipedia.org/wiki/Plasma_beta
    (refer to "K. STASIEWICZ1, 1999, SMALL SCALE ALFVÉNIC STRUCTURE IN THE AURORA" [1])
    low-beta: $\beta < m_e / m_i$ [1]
    intermediate-beta: $m_e / m_i < \beta < 1$ [1]
    :param n: number density (suppose $n=n_i=n_e$)
    :param T: plasma temperature (suppose $T=(T_i + T_e) / 2$) [1]
    :param B: measured magnetic field
    :return:
    """
    return (2 * PhysicalParameters.mu0 * PhysicalParameters.kB * n * T) / (B ** 2)


def get_complex_impedance(mu0, va, Sigma_P, omega, z):
    """
    use "np.abs()" to get the magnitude of complex impedance
    :param mu0:
    :param va:
    :param Sigma_P:
    :param omega: angular frequency
    :param z: the distance from the reflection point (usually 100~200 km altitude)
    :return:
    """
    Gamma = (1 / Sigma_P - mu0 * va) / (1 / Sigma_P + mu0 * va)
    return mu0 * va * ((1 + Gamma * np.exp(-2j * omega * z / va)) / (1 - Gamma * np.exp(-2j * omega * z / va)))


def E_B_ratio_kaw(va, f, rho_i, rho_s, lambda_e, v_fit):
    """
    refer to: HULL A J, CHASTON C C, DAMIANO P A. Multipoint Cluster Observations of Kinetic Alfvén Waves, Electron Energization, and O + Ion Outflow Response in the Mid‐Altitude Cusp Associated With Solar Wind Pressure and/or IMF B  Z  Variations[J/OL]. Journal of Geophysical Research: Space Physics, 2023, 128(11): e2023JA031982. DOI:10.1029/2023JA031982.
    :param va:
    :param f: frequency
    :param rho_i:
    :param rho_s:
    :param lambda_e:
    :param v_fit: the velocity corresponding to the fitted curve
    :return:
    """
    k_transverse = (2 * np.pi * f) / v_fit
    return va * np.sqrt(
        (1 + (k_transverse ** 2) * (lambda_e ** 2)) / (1 + (k_transverse ** 2) * (rho_i ** 2 + rho_s ** 2))) * (
            1 + (k_transverse ** 2) * (rho_i ** 2))


