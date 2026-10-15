# noether-ledger

Noether's theorem applied to a transformer, measured rather than argued.
A 1-layer, 1-head transformer with a ReLU MLP (numpy, hand-written backprop, gradcheck 4e-6) learns
Fibonacci mod 17 (x_t = x_{t-1} + x_{t-2}). It trains on 200 of the 289 possible sequences, and the other 89 are held out.

## Status: a replication, not a discovery

This repo is a correct, from-scratch, numerically verified reproduction of known results from roughly 2018–2022. Nothing here is new. Each part maps to existing work:

| result here | already known from |
|---|---|
| Weight-symmetry charges conserved under gradient flow; SGD, momentum, Adam and weight decay break them, including the finite-learning-rate (lr²) breaking that `check_identity.py` checks | Kunin et al., *Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics*, ICLR 2021 ([arXiv:2012.04728](https://arxiv.org/abs/2012.04728)) |
| Balancedness appearing without a regulariser | Du, Hu & Lee, *Algorithmic Regularization in Learning Deep Homogeneous Models*, NeurIPS 2018 |
| Exp 2: imbalanced (non-balanced gauge) init trains faster; the conserved imbalance sets the speed | Min, Tarmoun, Vidal & Mallada, ICML 2021 ([arXiv:2105.06351](https://arxiv.org/abs/2105.06351)) |
| Using the gauge freedom deliberately to speed up training | Zhao et al., *Symmetry Teleportation for Accelerated Optimization*, 2022 |
| Charge drift as a learning-event detector | Not new in substance. By the identity, drift per step = lr² × a difference of gradient Gram matrices, so it is a gradient-norm signal in disguise, and gradient norms spiking at phase transitions (e.g. grokking) is well known |
| Exp 3: measuring how equivariant a trained network is | Gruver et al., *The Lie Derivative for Measuring Learned Equivariance*, 2022; small transformers on modular arithmetic: Nanda et al. on grokking, 2023 |

The repo started from a question prompted by an Instagram post about Emmy Noether ("is there token symmetry?"). It is kept as a worked, runnable illustration of what Noether's theorem does and does not mean inside a neural network.

## Weight space: real Noether charges

Three continuous symmetries leave the network function unchanged:

| symmetry | transform | charge |
|---|---|---|
| Q/K gauge | W_Q→W_Q A, W_K→W_K A⁻ᵀ | C_QK = W_QᵀW_Q − W_KᵀW_K |
| V/O gauge | W_V→W_V B, W_O→B⁻¹W_O | C_VO = W_VᵀW_V − W_OW_Oᵀ |
| ReLU rescale | w_in,j·α, w_out,j/α | c_j = ‖w_in,j‖² − ‖w_out,j‖² |

**Exact discrete law (check_identity.py).** For plain SGD, the symmetry forces W_QᵀG_Q = G_KᵀW_K. The O(lr) change in the charge therefore cancels identically (measured 2e-15), and each step changes it by exactly

  C(t+1) − C(t) = lr² (G_QᵀG_Q − G_KᵀG_K)

This was verified to 5e-8 (QK), 1e-9 (VO) and 2e-9 (MLP). Gradient flow conserves the charge exactly. SGD leaks only at order lr².

**Exp 1 (charges.png).** Relative drift of each charge after 3000 steps:

| optimizer | held-out acc | QK drift | VO drift | MLP drift |
|---|---|---|---|---|
| SGD lr=1e-2 (does not learn) | 0.02 | 2e-7 | 3e-6 | 2e-6 |
| SGD lr=1e-1 | 0.98 | 2e-4 | 3e-2 | 1e-2 |
| SGD lr=3e-1 | 1.00 | 1e-3 | 9e-2 | 5e-2 |
| SGD+momentum lr=3e-2 | 0.98 | 9e-3 | 4e-3 | 4e-3 |
| Adam lr=3e-3 | 0.94 | 0.13 | 0.13 | 0.09 |
| AdamW lr=3e-3 wd=0.1 | 0.99 | 0.16 | 0.27 | 0.24 |

- Adam breaks the symmetry at first order: about 10% of the charge leaks within the first ~20 steps, and the leak then nearly stops.
- Under SGD the charge is fixed while the weights grow. The *relative* imbalance |C|/scale therefore falls (0.43 → 0.22), and balancedness appears with no regulariser.
- **The leak marks the learning event.** The SGD drift curves jump in steps, and each jump coincides with the jump in held-out accuracy (step ~1700 at lr=1e-1, ~450 at lr=3e-1). The leak per step is lr²·|gradient charge|, which is large only while the network is reorganising, so charge drift works as a label-free detector of the moment it learns.

**Exp 2 (gauge_init.png).** The init is scaled by W_Q·a, W_K/a, W1·a, W2/a. At init the logits are bit-identical (max diff 0), so this is the same function with a different charge.

| a | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| SGD held-out acc @1500 | 0.87 | 0.66 | **0.40** | 0.55 | 0.72 |
| Adam held-out acc @1500 | 0.96 | 0.97 | 0.93 | 0.89 | 0.80 |

Under SGD the gauge is invisible to the function but not to learning. The balanced start (a=1) was the *slowest*. a and 1/a give nearly identical curves, so what matters is the size of the conserved imbalance, not its sign. Because SGD can never change the charge, the init gauge acts as a permanent per-layer learning-rate setting. Adam is insensitive in speed, but its generalisation still varies with a.

## Token space: discrete symmetries (invariances, not conservation laws)

**Exp 3 (exp3_token_ledger.py).** For each transform g, the ledger measures KL(p(·|x) mapped by g ‖ p(·|g x)) and argmax agreement over all 289 sequences.

| transform | kind | SGD 0.3 (held 1.00) | AdamW (0.99) | SGD 0.1 (0.98) | SGD+mom (0.98) | Adam (0.945) |
|---|---|---|---|---|---|---|
| residual basis change GL(d), QK gauge, ReLU rescale | exact | ~1e-17 | ~1e-17 | ~1e-17 | ~1e-17 | ~1e-17 |
| swap last two tokens (a+b=b+a) | task | 1.0e-2 | 4.7e-3 | 0.10 | 0.12 | 0.62 |
| relabel x→3x mod 17 | task | 3.6e-2 | 3.5e-2 | 0.24 | 0.20 | 0.44 |
| time shift by 5 | task | 1.2e-2 | 1.9e-2 | 0.07 | 0.08 | 0.21 |
| shuffle history older than 2 tokens | probe | 7.6 (agree 0.60) | 7.8 | 7.4 | 11 | 13 |
| add 5 to every token | control | 15 | – | – | – | 22 |

- The exact gauges come out at machine zero, which confirms the ledger itself has no bugs.
- **The violation of the task symmetries tracks generalisation, not the optimizer.** AdamW breaks the weight charges the most and still learned commutativity best. The ledger needs no labels, so it is a candidate label-free generalisation meter.
- None of the models is invariant to shuffling the older history. They all read tokens beyond the last two. On real data that history is redundant (any two Fibonacci terms fix the rest), so this is a fact about the learned *mechanism* that accuracy cannot reveal.

## Honest limits

See "Status" above: the results are known, not new. On top of that, these results come from one task, one architecture and one seed per config, so the n=5 correlation in exp 3 is suggestive only. Exp 2 compares runs at a fixed step budget, not at convergence. Transformers have no action, so the charges belong to the optimizer flow, not to the forward pass.

## Run

```
python3 check_identity.py; python3 exp1_charges.py 3000; python3 plot1.py
python3 exp2_gauge_init.py; python3 plot2.py
python3 exp3_token_ledger.py params_sgd_0.3.npz
```
Requires numpy and matplotlib only.
