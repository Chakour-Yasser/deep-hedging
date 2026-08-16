"""Test de sante fondamental : Heston avec vol-of-vol nulle EST un GBM.

Si xi = 0 et v0 = theta, la variance reste constante egale a theta, donc Heston
se reduit exactement au brownien geometrique de vol sqrt(theta). En pilotant les
deux simulateurs avec le meme bruit, les trajectoires doivent coincider a la
precision machine. Ce test attrape la quasi-totalite des bugs de simulation.
"""
import numpy as np
from deephedge.simulate import simulate_gbm, simulate_heston


def test_heston_xi0_equals_gbm():
    S0, mu, theta, T = 100.0, 0.05, 0.04, 1.0
    n_steps, n_paths = 100, 5000
    rng = np.random.default_rng(0)
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))

    gbm = simulate_gbm(S0, mu, np.sqrt(theta), T, n_steps, n_paths, z=z1)
    heston, v = simulate_heston(S0, theta, mu, kappa=2.0, theta=theta, xi=0.0,
                                rho=-0.7, T=T, n_steps=n_steps, n_paths=n_paths,
                                z1=z1, z2=z2)

    assert np.allclose(v, theta), "la variance doit rester constante quand xi = 0"
    max_diff = np.abs(gbm - heston).max()
    assert max_diff < 1e-9, f"ecart max {max_diff:.2e}, attendu ~0"


if __name__ == "__main__":
    test_heston_xi0_equals_gbm()
    print("OK : Heston(xi=0, v0=theta) == GBM(sigma=sqrt(theta))")
