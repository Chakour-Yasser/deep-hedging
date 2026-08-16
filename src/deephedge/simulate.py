"""Simulateurs de marche : brownien geometrique et Heston.

Le GBM utilise la solution exacte (log-normale), donc pas d'erreur de schema.
Heston utilise Euler full-truncation (la variance est tronquee a zero a chaque
pas), un schema simple et stable pour un modele a variance CIR.
"""
import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, rng=None, z=None):
    """Trajectoires de brownien geometrique, forme (n_paths, n_steps + 1).

    dS = mu S dt + sigma S dW, solution exacte
        S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z).

    z : tirages standard normaux (n_paths, n_steps) a fournir pour des nombres
    aleatoires communs (reproductibilite, tests d'equivalence).
    """
    rng = rng or np.random.default_rng(0)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps)) if z is None else np.asarray(z)
    incr = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    logpaths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(logpaths)


def simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, n_steps, n_paths, rng=None,
                    z1=None, z2=None):
    """Trajectoires Heston (Euler full-truncation). Renvoie (S, v).

    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,  corr(dW1, dW2) = rho.
    theta est la VARIANCE de long terme (sqrt(theta) est la vol).

    z1, z2 : tirages standard normaux INDEPENDANTS (n_paths, n_steps) a fournir
    pour des nombres aleatoires communs. La correlation rho est appliquee en
    interne : dW2 = rho z1 + sqrt(1 - rho^2) z2.
    """
    rng = rng or np.random.default_rng(0)
    dt = T / n_steps
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    for k in range(n_steps):
        Z1 = rng.standard_normal(n_paths) if z1 is None else np.asarray(z1)[:, k]
        Z2b = rng.standard_normal(n_paths) if z2 is None else np.asarray(z2)[:, k]
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2b
        vk = np.maximum(v[:, k], 0.0)
        v[:, k + 1] = np.maximum(v[:, k] + kappa * (theta - vk) * dt
                                 + xi * np.sqrt(vk * dt) * Z2, 0.0)
        S[:, k + 1] = S[:, k] * np.exp((mu - 0.5 * vk) * dt + np.sqrt(vk * dt) * Z1)
    return S, v
