# Deep Hedging

I built a neural network that learns to hedge options by minimizing tail risk under
transaction costs, instead of following a closed-form delta. The project starts from a
price simulator and ends at a bid-ask quote, one step at a time. Every result is checked
against a benchmark on the same simulated paths, so the numbers below are comparisons, not
absolute claims.

## The idea

Classical hedging assumes you can trade continuously with no cost, so the delta from
Black-Scholes replicates the option perfectly. Real desks trade a few times a day and pay
a spread on every trade, so perfect replication is impossible and what is left is risk.
Deep hedging reframes the problem: instead of computing a delta, I parametrize the hedging
position as a neural network of the current state and train it to minimize a risk measure
of the final hedged P&L. The risk measure I use is the CVaR at 95%, the average of the
worst 5% outcomes, because on an incomplete market you care about the tail, not the
variance.

The same trained network also prices the option. Because CVaR is cash-invariant, the
smallest premium that makes a hedged short position acceptable is just the discounted
residual risk of the best hedge, which is exactly the quantity the network minimizes. So
the hedger is the pricer, and the bid-ask spread is the price of the risk you cannot hedge
away. There is a short note with the full derivation in `docs/`.

## What I found

On a Black-Scholes market with no cost, the network reproduces the delta hedge, which is
the reassuring sanity check: when replication is possible, it finds it. With transaction
costs it matches the best no-trade band, so it learns to trade less when trading is
expensive.

On a Heston market, where volatility is stochastic and the market is incomplete, it beats
the delta. The CVaR of the hedged position drops from 4.57 (delta) to 3.86 (network),
about 16% less tail risk on the same paths. With transaction costs added, it reaches 6.90
against 8.05 for the best delta-with-band, about 14% less.

I added a second option to hedge the vega, and the CVaR falls to 2.45. I did not stop at
that number, because it flatters the method. A static hedge of one option already gets to
3.90, and the textbook delta-vega hedge oversells by trading too much (6.54). The honest
statement is that the dynamic network improves 3.90 to 2.45, not 6.54 to 2.45.

The hardest case is an up-and-out barrier option. Here the classical delta from the
barrier formula is worse than doing nothing (26.5 versus 20.7), because the delta explodes
near the barrier and the fixed grid cannot follow it. The network gets 4.96, against 6.44
for a static vanilla hedge. Getting there took three failed training runs, all traced back
to the loss: the Rockafellar-Uryasev form of CVaR was stalling under Adam, and the fix was
a direct empirical CVaR plus a residual head anchored on the static hedge plus fresh Monte
Carlo each epoch.

I then tested robustness by training on one model and evaluating on another. The
conclusion is that robustness is a property of the product and of what the network
observes, not of the method itself. A vanilla generalizes on its own, but the barrier is
fragile when the network cannot see the volatility, because the knock-out probability
triples across regimes.

## What is in here

The story is in `notebooks/00_the_whole_story.ipynb`, which runs end to end and carries
the narrative. The numbered notebooks are the individual chapters where each piece was
built. The reusable code lives in `src/deephedge` (path simulation, pricers, losses,
hedgers, indifference pricing) with tests in `tests/`, including one that checks the Heston
pricer collapses to Black-Scholes when vol-of-vol is zero, to nine decimals.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/
jupyter notebook notebooks/00_the_whole_story.ipynb
```

The training cells use PyTorch and are meant to run on a machine with a GPU or a recent
laptop. The numerical logic outside training is plain NumPy.

## Standard parameters

S0 = K = 100, mu = 0.05, r = 0.02, T = 1, 63 rebalancing steps, proportional cost 0.01,
CVaR level 0.95. Heston: v0 = theta = 0.04, kappa = 2, xi = 0.3, rho = -0.7.

## Notes to myself

The recurring lesson was that the loss matters more than the architecture. Most of the time
I lost was to a CVaR objective that silently optimized the mean, and the wins came from
fixing what the network was actually being asked to minimize, not from making it bigger.
