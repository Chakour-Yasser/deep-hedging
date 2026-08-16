"""Verifications des pricers contre Monte Carlo et contre des identites connues."""
import numpy as np
from deephedge.pricers import (bs_call, heston_call, bs_up_and_out_call,
                               digital_call)
from deephedge.simulate import simulate_gbm, simulate_heston


PARAMS = dict(r=0.02, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


def test_heston_call_matches_mc():
    S0, v0, T = 100.0, 0.04, 1.0
    rng = np.random.default_rng(1)
    S, _ = simulate_heston(S0, v0, PARAMS["r"], PARAMS["kappa"], PARAMS["theta"],
                           PARAMS["xi"], PARAMS["rho"], T, 200, 200_000, rng=rng)
    for K in (90.0, 100.0, 110.0):
        cf = heston_call(S0, v0, PARAMS["r"], PARAMS["kappa"], PARAMS["theta"],
                         PARAMS["xi"], PARAMS["rho"], T, K)
        mc = np.exp(-PARAMS["r"] * T) * np.maximum(S[:, -1] - K, 0.0).mean()
        assert abs(cf - mc) < 0.05, f"K={K}: CF {cf:.3f} vs MC {mc:.3f}"


def test_barrier_matches_mc_with_continuity_correction():
    S0, K, H, T, r, sig = 100.0, 100.0, 130.0, 1.0, 0.02, 0.20
    n = 63
    rng = np.random.default_rng(2)
    S = simulate_gbm(S0, r, sig, T, n, 200_000, rng=rng)
    knock = S.max(axis=1) >= H
    mc = np.exp(-r * T) * np.where(~knock, np.maximum(S[:, -1] - K, 0.0), 0.0).mean()
    Hd = H * np.exp(0.5826 * sig * np.sqrt(T / n))          # Broadie-Glasserman-Kou
    cf = float(bs_up_and_out_call(S0, K, Hd, T, r, sig))
    assert abs(cf - mc) < 0.05, f"barriere CF {cf:.3f} vs MC {mc:.3f}"


def test_heston_reduces_to_bs_when_almost_flat():
    # xi tres petit, v0=theta : Heston tend vers BS de vol sqrt(theta).
    # (xi=0 exactement est degenere pour le pricer de Fourier, division par xi^2.)
    S0, K, T, r, theta = 100.0, 100.0, 1.0, 0.02, 0.04
    cf = heston_call(S0, theta, r, 2.0, theta, 1e-3, -0.7, T, K)
    bs = float(bs_call(S0, K, T, r, np.sqrt(theta)))
    assert abs(cf - bs) < 5e-2, f"Heston quasi-plat {cf:.4f} vs BS {bs:.4f}"


def test_digital_below_one():
    # une digitale cash-or-nothing vaut entre 0 et exp(-rT)
    p = float(digital_call(100.0, 100.0, 1.0, 0.02, 0.20))
    assert 0.0 < p < 1.0


if __name__ == "__main__":
    test_heston_call_matches_mc()
    test_barrier_matches_mc_with_continuity_correction()
    test_heston_reduces_to_bs_when_almost_flat()
    test_digital_below_one()
    print("OK : pricers coherents avec Monte Carlo et identites connues")
