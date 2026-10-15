"""Exact discrete Noether law for SGD: C(t+1)-C(t) = lr^2 (G_Q^T G_Q - G_K^T G_K), likewise VO and MLP.
The O(lr) term cancels exactly because of the symmetry; only O(lr^2) survives."""
import numpy as np, model as M
xtr, _ = M.split(); p = M.init(); lr = 0.1; worst = {}
for s in range(40):
    _, G = M.loss_grad(p, xtr); c = M.charges(p)
    pred = dict(QK=lr**2 * (G['WQ'].T @ G['WQ'] - G['WK'].T @ G['WK']),
                VO=lr**2 * (G['WV'].T @ G['WV'] - G['WO'] @ G['WO'].T),
                MLP=lr**2 * ((G['W1']**2).sum(0) - (G['W2']**2).sum(1)))
    first = dict(QK=-lr * (p['WQ'].T @ G['WQ'] - G['WK'].T @ p['WK']))  # symmetry => W_Q^T G_Q = G_K^T W_K
    for k in p: p[k] -= lr * G[k]
    c2 = M.charges(p)
    for q in c: worst[q] = max(worst.get(q, 0), np.linalg.norm(c2[q] - c[q] - pred[q]) / np.linalg.norm(pred[q]))
    worst['QK first-order term / its parts'] = max(worst.get('QK first-order term / its parts', 0),
        np.linalg.norm(first['QK']) / (lr * np.linalg.norm(p['WQ'].T @ G['WQ'])))
for k, v in worst.items(): print(f"{k:32s} worst relative mismatch over 40 steps: {v:.1e}")
