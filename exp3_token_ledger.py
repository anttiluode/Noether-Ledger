"""Exp 3: forward-pass symmetry ledger. For each transformation g, measure how far the model is
from equivariance: mean KL( p(.|x) mapped by g  ||  p(.|g x) ) over held-out + train sequences.
Exact gauges should give ~0 (bug detector); task symmetries are either learned or not; controls break."""
import numpy as np, sys, model as M
X = M.all_sequences(); xtr, xte = M.split(); V, T = M.V, M.T
def lp(p, x): return M.logp(M.fwd(p, x)['lg'])
def kl(a, b): return float((np.exp(a) * (a - b)).sum(-1).mean())
def agree(a, b): return float((a.argmax(-1) == b.argmax(-1)).mean())
rng = np.random.default_rng(7)
def ledger(p):
    rows = []; L0 = lp(p, X); pos = slice(1, T - 1)  # positions where the next token is determined
    # --- exact weight gauges (forward pass must be unchanged) ---
    R = rng.standard_normal((M.d, M.d)); Ri = np.linalg.inv(R)
    q = dict(p); q['E'] = p['E'] @ R; q['P'] = p['P'] @ R
    for k in ['WQ', 'WK', 'WV', 'W1']: q[k] = Ri @ p[k]
    q['WO'] = p['WO'] @ R; q['W2'] = p['W2'] @ R; q['U'] = Ri @ p['U']
    rows.append(('exact', 'residual basis change R in GL(d)', L0, lp(q, X)))
    A = rng.standard_normal((M.dh, M.dh)); q = dict(p); q['WQ'] = p['WQ'] @ A; q['WK'] = p['WK'] @ np.linalg.inv(A).T
    rows.append(('exact', 'Q/K gauge A in GL(dh)', L0, lp(q, X)))
    s = np.exp(rng.standard_normal(M.dm)); q = dict(p); q['W1'] = p['W1'] * s; q['W2'] = p['W2'] / s[:, None]
    rows.append(('exact', 'ReLU unit rescaling', L0, lp(q, X)))
    # --- task symmetries (data is invariant/equivariant; did the model learn it?) ---
    for c in [3, 6]:
        sig = (c * np.arange(V)) % V  # Fibonacci mod 17 is closed under x -> c x
        Lg = lp(p, sig[X]); rows.append(('task', f'relabel x -> {c}x mod 17 (equivariant)', L0[:, pos][..., sig.argsort()].copy() if False else L0[:, pos], Lg[:, pos][..., sig]))
    # translation: sequence x shifted by k is again a Fibonacci sequence; pred at t on x == pred at t-k on shifted
    ks = []
    for k in [1, 3, 5]:
        idx = X[:, k] * V + X[:, k + 1]; Xs = X[idx]; Ls = lp(p, Xs)
        ks.append((f'time shift by {k} (re-start sequence later)', L0[:, k + 1:T - 1], Ls[:, 1:T - 1 - k]))
    for name, a, b in ks: rows.append(('task', name, a, b))
    # swap the two most recent tokens: next token a+b is symmetric in (a,b)
    A_, B_ = [], []
    for t in range(2, T - 1):
        Xw = X.copy(); Xw[:, [t - 1, t]] = Xw[:, [t, t - 1]]; A_.append(L0[:, t]); B_.append(lp(p, Xw)[:, t])
    rows.append(('task', 'swap last two tokens (a+b=b+a)', np.stack(A_, 1), np.stack(B_, 1)))
    # shuffle the older history: next token only depends on the last two
    A_, B_ = [], []
    for t in range(4, T - 1):
        Xw = X.copy(); perm = rng.permutation(t - 1); Xw[:, :t - 1] = X[:, perm]; A_.append(L0[:, t]); B_.append(lp(p, Xw)[:, t])
    rows.append(('task', 'shuffle history older than 2 tokens', np.stack(A_, 1), np.stack(B_, 1)))
    # --- controls: NOT symmetries of the task ---
    Lg = lp(p, (X + 5) % V); sig = (np.arange(V) + 5) % V
    rows.append(('control', 'add 5 to every token (not a symmetry)', L0[:, pos], Lg[:, pos][..., sig]))
    A_, B_ = [], []
    for t in range(2, T - 1):
        Xw = X.copy(); Xw[:, t - 1] = rng.integers(0, V, len(X)); A_.append(L0[:, t]); B_.append(lp(p, Xw)[:, t])
    rows.append(('control', 'resample 2nd-last token (not a symmetry)', np.stack(A_, 1), np.stack(B_, 1)))
    return [(cat, name, kl(a, b), agree(a, b)) for cat, name, a, b in rows]
if __name__ == '__main__':
    trained = dict(np.load(sys.argv[1])); init = M.init()
    print(f"trained model acc train/held-out: {M.accuracy(trained, xtr):.3f}/{M.accuracy(trained, xte):.3f}")
    lt, li = ledger(trained), ledger(init)
    print(f"{'kind':8s} {'transformation':44s} {'KL init':>9s} {'KL trained':>11s} {'argmax agree (trained)':>23s}")
    for (c, n, k1, a1), (_, _, k0, _) in zip(lt, li): print(f"{c:8s} {n:44s} {k0:9.2e} {k1:11.2e} {a1:23.3f}")
