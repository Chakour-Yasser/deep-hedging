"""Verifications des mesures de risque."""
import numpy as np
from deephedge.losses import cvar, rho_lambda


def test_cvar_matches_definition():
    # pour une gaussienne, CVaR_0.95 de la perte = mu + sigma * phi(z)/(1-a)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2_000_000)          # pertes standard normales
    a = 0.95
    from scipy.stats import norm
    theo = norm.pdf(norm.ppf(a)) / (1 - a)      # ~2.063
    emp = cvar(x, a, is_loss=True)
    assert abs(emp - theo) < 0.02, f"CVaR empirique {emp:.3f} vs theorie {theo:.3f}"


def test_cvar_translation_invariance():
    rng = np.random.default_rng(1)
    L = rng.standard_normal(100_000)
    c = 3.7
    assert abs(cvar(L + c, 0.95, is_loss=True) - (cvar(L, 0.95, is_loss=True) + c)) < 1e-9


def test_rho_lambda_between_mean_and_cvar():
    rng = np.random.default_rng(2)
    L = rng.standard_normal(200_000)
    m = L.mean()
    c = cvar(L, 0.95, is_loss=True)
    for lam in (0.0, 0.3, 0.7, 1.0):
        r = rho_lambda(L, 0.95, lam)
        assert m - 1e-6 <= r <= c + 1e-6, f"lam={lam}: rho {r:.3f} hors [{m:.3f}, {c:.3f}]"


if __name__ == "__main__":
    test_cvar_matches_definition()
    test_cvar_translation_invariance()
    test_rho_lambda_between_mean_and_cvar()
    print("OK : mesures de risque conformes")
