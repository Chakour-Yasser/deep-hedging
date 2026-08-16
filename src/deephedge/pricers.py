"""Pricers analytiques et semi-analytiques.

- Black-Scholes (call, delta) et digitale cash-or-nothing.
- Call up-and-out (Reiner-Rubinstein, monitoring continu).
- Heston par fonction caracteristique (inversion de Fourier, forme a deux
  probabilites), verifie contre Monte Carlo dans les tests.
- Tabulation du pricer Heston sur une grille (S, v, tau) pour l'interpolation
  rapide utilisee par les hedgers (le pricer exact est trop lent en boucle).
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


# ------------------------- Black-Scholes -------------------------------------
def bs_call(S, K, tau, r, sigma):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_call_delta(S, K, tau, r, sigma):
    S = np.asarray(S, float)
    return norm.cdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)))


def digital_call(S, K, tau, r, sigma):
    """Cash-or-nothing : paie 1 si S_T >= K."""
    S = np.asarray(S, float)
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return np.exp(-r * tau) * norm.cdf(d2)


def digital_call_delta(S, K, tau, r, sigma):
    S = np.asarray(S, float)
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return np.exp(-r * tau) * norm.pdf(d2) / (S * sigma * np.sqrt(tau))


def bs_up_and_out_call(S, K, H, tau, r, sigma):
    """Call up-and-out (strike K < barriere H), monitoring continu.

    Formule de Reiner-Rubinstein : c_uo = A - B + C - D (phi=1, eta=-1).
    Pour un monitoring discret a n pas, appliquer la correction de continuite
    de Broadie-Glasserman-Kou en decalant H -> H exp(0.5826 sigma sqrt(T/n)).
    """
    S = np.asarray(S, float)
    srt = sigma * np.sqrt(tau)
    mu = (r - 0.5 * sigma ** 2) / sigma ** 2
    x1 = np.log(S / K) / srt + (1 + mu) * srt
    x2 = np.log(S / H) / srt + (1 + mu) * srt
    y1 = np.log(H ** 2 / (S * K)) / srt + (1 + mu) * srt
    y2 = np.log(H / S) / srt + (1 + mu) * srt
    A = S * norm.cdf(x1) - K * np.exp(-r * tau) * norm.cdf(x1 - srt)
    B = S * norm.cdf(x2) - K * np.exp(-r * tau) * norm.cdf(x2 - srt)
    C = S * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y1) \
        - K * np.exp(-r * tau) * (H / S) ** (2 * mu) * norm.cdf(-y1 + srt)
    D = S * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y2) \
        - K * np.exp(-r * tau) * (H / S) ** (2 * mu) * norm.cdf(-y2 + srt)
    return np.where(S >= H, 0.0, A - B + C - D)


# ------------------------- Heston (fonction caracteristique) -----------------
def _heston_cf(phi, S0, v0, r, kappa, theta, xi, rho, T):
    out = []
    for u, b in [(0.5, kappa - rho * xi), (-0.5, kappa)]:
        d = np.sqrt((rho * xi * 1j * phi - b) ** 2 - xi ** 2 * (2 * u * 1j * phi - phi ** 2))
        g = (b - rho * xi * 1j * phi + d) / (b - rho * xi * 1j * phi - d)
        C = r * 1j * phi * T + (kappa * theta / xi ** 2) * (
            (b - rho * xi * 1j * phi + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
        D = (b - rho * xi * 1j * phi + d) / xi ** 2 * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        out.append(np.exp(C + D * v0 + 1j * phi * np.log(S0)))
    return out


def heston_call(S0, v0, r, kappa, theta, xi, rho, T, K):
    """Prix d'un call europeen sous Heston par inversion de Fourier."""
    def integ(phi, i):
        f = _heston_cf(phi, S0, v0, r, kappa, theta, xi, rho, T)[i]
        return (np.exp(-1j * phi * np.log(K)) * f / (1j * phi)).real
    P1 = 0.5 + quad(integ, 1e-8, 200, args=(0,), limit=200)[0] / np.pi
    P2 = 0.5 + quad(integ, 1e-8, 200, args=(1,), limit=200)[0] / np.pi
    return S0 * P1 - K * np.exp(-r * T) * P2


def build_heston_grid(Sg, vg, tg, r, kappa, theta, xi, rho, K):
    """Tabule heston_call sur la grille (Sg, vg, tg). Renvoie un tableau 3D.

    A brancher sur scipy.interpolate.RegularGridInterpolator pour une lecture en
    microsecondes au lieu de quelques millisecondes par appel du pricer exact.
    """
    grid = np.zeros((len(Sg), len(vg), len(tg)))
    for i, S in enumerate(Sg):
        for j, v in enumerate(vg):
            for k, tau in enumerate(tg):
                grid[i, j, k] = heston_call(S, v, r, kappa, theta, xi, rho, tau, K)
    return grid
