"""Pricing par indifference : le hedger est un pricer.

Le prix vendeur est la prime minimale telle que, apres couverture optimale, le
risque de la position n'est pas pire que ne rien faire. Avec une mesure de risque
invariante par translation (CVaR, ou rho_lambda), ca se resout en fermé :

    p_ask = exp(-rT) * rho( payoff - gains )      [position vendue, prime nulle]
    p_bid = -exp(-rT) * rho( -gains - payoff )    [position achetee, prime nulle]
    spread = p_ask - p_bid = exp(-rT) * (rho(L_ask) + rho(L_bid)).

Sans friction ni incompletude, les risques residuels s'annulent et bid = ask =
prix risque-neutre. Le spread mesure donc la difficulte de couverture du produit.
"""
import numpy as np
from .losses import rho_lambda


def ask_price(loss_short, r, T, alpha=0.95, lam=1.0):
    """loss_short = payoff - gains de la position vendue (prime nulle)."""
    return np.exp(-r * T) * rho_lambda(loss_short, alpha, lam)


def bid_price(loss_long, r, T, alpha=0.95, lam=1.0):
    """loss_long = -gains - payoff de la position achetee (prime nulle)."""
    return -np.exp(-r * T) * rho_lambda(loss_long, alpha, lam)


def spread(loss_short, loss_long, r, T, alpha=0.95, lam=1.0):
    return ask_price(loss_short, r, T, alpha, lam) - bid_price(loss_long, r, T, alpha, lam)
