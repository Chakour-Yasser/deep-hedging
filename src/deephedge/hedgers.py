"""Couvreurs : reseau de deep hedging (PyTorch) et benchmarks classiques (numpy).

Le reseau apprend une politique (etat -> position) qui minimise le CVaR de la
perte couverte sous couts. Le marche est exogene (nos trades n'impactent pas les
prix), donc on peut simuler les trajectoires et les prix d'instruments comme des
donnees figees ; le gradient ne passe que par les actions du reseau.
"""
import numpy as np


# ----------------------------- benchmark classique ---------------------------
def band_delta_hedge(S, payoff, delta_fn, premium, r, dt, cost, band=0.0, gate=None):
    """Delta-hedge a bande de non-trading. Renvoie le P&L par trajectoire.

    delta_fn(S_k, tau) donne la position cible. band=0 est le delta pur ;
    une bande > 0 saute les petits ajustements pour economiser les couts.
    gate (forme comme S) desactive la position une fois une barriere franchie.
    """
    m, n1 = S.shape
    n = n1 - 1
    T = n * dt
    times = np.linspace(0, T, n1)
    cash = np.full(m, premium)
    pos = np.zeros(m)
    for k in range(n):
        tau = max(T - times[k], 1e-4)
        g = 1.0 if gate is None else gate[:, k]
        tgt = delta_fn(S[:, k], tau) * g
        tr = np.where(np.abs(tgt - pos) > band, tgt - pos, 0.0)
        cash -= tr * S[:, k] + cost * np.abs(tr) * S[:, k]
        pos += tr
        cash *= np.exp(r * dt)
    return cash + pos * S[:, -1] - payoff


# ----------------------------- reseau (PyTorch) ------------------------------
def make_net(in_dim, hidden=32, out_dim=1, zero_last=False):
    """MLP simple. zero_last=True initialise la derniere couche a zero, utile
    pour le hedging residuel (la politique de depart vaut l'ancre choisie)."""
    import torch
    net = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_dim))
    if zero_last:
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)
    return net


def hedge_pnl_torch(net, S, feats_fn, payoff, premium, r, dt, cost):
    """P&L couvert d'un reseau mono-instrument le long de S (tenseur figé).

    feats_fn(S, k, pos) renvoie le tenseur d'etat au pas k (forme (m, in_dim)).
    payoff est un tenseur (m,), calcule en amont depuis S.
    """
    import torch
    m, n1 = S.shape
    n = n1 - 1
    cash = torch.full((m,), float(premium))
    pos = torch.zeros(m)
    for k in range(n):
        d = net(feats_fn(S, k, pos)).squeeze(-1)
        tr = d - pos
        cash = cash - tr * S[:, k] - cost * torch.abs(tr) * S[:, k]
        cash = cash * float(np.exp(r * dt))
        pos = d
    return cash + pos * S[:, -1] - payoff


def train_hedger(sample_batch, feats_fn, payoff_fn, in_dim, premium, r, dt, cost,
                 epochs=400, lr=1e-3, alpha=0.95, hidden=32, zero_last=False, seed=0):
    """Entraine un hedger mono-instrument avec perte CVaR empirique directe.

    sample_batch() renvoie un tenseur S (m, n+1) frais a chaque appel (Monte
    Carlo frais, pas de jeu fige a memoriser). payoff_fn(S) -> tenseur (m,).
    """
    import torch
    from .losses import cvar_torch
    torch.manual_seed(seed)
    net = make_net(in_dim, hidden, 1, zero_last)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        S = sample_batch().detach()
        pnl = hedge_pnl_torch(net, S, feats_fn, payoff_fn(S), premium, r, dt, cost)
        loss = cvar_torch(-pnl, alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net
