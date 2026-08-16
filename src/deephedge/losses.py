"""Mesures de risque servant de perte a l'entrainement.

Lecon apprise a la dure (voir notebook 13) : pour minimiser un CVaR par descente
de gradient, la forme de Rockafellar-Uryasev avec variable auxiliaire w est
piegeuse. Adam deplace un scalaire d'environ le learning rate par pas (il
normalise le gradient), donc w reste coince pres de 0, et l'objectif devient la
perte MOYENNE au lieu de la queue. On prefere le CVaR empirique direct, qui cible
exactement la queue qu'on reporte.
"""
import numpy as np


def cvar(pnl_or_loss, alpha=0.95, is_loss=False):
    """CVaR au niveau alpha.

    Par defaut l'argument est un P&L (on prend la perte = -pnl). Passer
    is_loss=True si on fournit deja une perte.
    """
    loss = pnl_or_loss if is_loss else -np.asarray(pnl_or_loss)
    var = np.quantile(loss, alpha)
    return loss[loss >= var].mean()


def rho_lambda(loss, alpha=0.95, lam=1.0):
    """Mesure de risque coherente rho = (1 - lam) E[L] + lam CVaR_alpha(L).

    lam est l'aversion au risque : lam = 1 donne le CVaR plein (pricing d'une
    transaction isolee, tres prudent), lam petit decrit un desk diversifie qui
    ne facture que le cout espere plus une petite marge.
    """
    loss = np.asarray(loss)
    return (1 - lam) * loss.mean() + lam * cvar(loss, alpha, is_loss=True)


def cvar_torch(loss, alpha=0.95):
    """CVaR empirique direct, differentiable pour l'entrainement PyTorch.

    Le sous-gradient passe par les elements de la queue selectionnes ; stable
    des qu'un lot donne une queue de quelques centaines de trajectoires.
    """
    import torch
    var = torch.quantile(loss, alpha)
    return loss[loss >= var].mean()
