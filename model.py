"""Tiny 1-layer transformer (1 head + ReLU MLP) in numpy with hand-written backprop.
Task: next token of a Fibonacci sequence mod 17: x_t = x_{t-1} + x_{t-2} (mod 17)."""
import numpy as np
V, T, d, dh, dm = 17, 12, 48, 24, 96

def all_sequences():
    a, b = np.meshgrid(np.arange(V), np.arange(V), indexing='ij')
    x = np.zeros((V * V, T), int); x[:, 0] = a.ravel(); x[:, 1] = b.ravel()
    for t in range(2, T): x[:, t] = (x[:, t-1] + x[:, t-2]) % V
    return x

def split(seed=0, n_train=200):
    x = all_sequences(); idx = np.random.default_rng(seed).permutation(len(x))
    return x[idx[:n_train]], x[idx[n_train:]]

def init(seed=0):
    r = np.random.default_rng(seed); g = lambda *s: r.standard_normal(s) / np.sqrt(s[0])
    return dict(E=g(V, d), P=g(T, d), WQ=g(d, dh), WK=g(d, dh), WV=g(d, dh), WO=g(dh, d),
                W1=g(d, dm), W2=g(dm, d), U=g(d, V))

MASK = np.triu(np.ones((T, T), bool), 1)

def fwd(p, x, P=None):
    h0 = p['E'][x] + (p['P'] if P is None else P)
    q, k, v = h0 @ p['WQ'], h0 @ p['WK'], h0 @ p['WV']
    s = q @ k.transpose(0, 2, 1) / np.sqrt(dh); s = np.where(MASK, -1e30, s); s = s - s.max(-1, keepdims=True)
    a = np.exp(s); a /= a.sum(-1, keepdims=True)
    o = a @ v; h1 = h0 + o @ p['WO']
    z = h1 @ p['W1']; m = np.maximum(z, 0); h2 = h1 + m @ p['W2']
    return dict(h0=h0, q=q, k=k, v=v, a=a, o=o, h1=h1, z=z, m=m, h2=h2, lg=h2 @ p['U'])

def logp(lg):
    lg = lg - lg.max(-1, keepdims=True); return lg - np.log(np.exp(lg).sum(-1, keepdims=True))

def loss_grad(p, x, need_grad=True):
    c = fwd(p, x); lp = logp(c['lg'][:, :-1]); y = x[:, 1:]; M = y.size
    L = -np.take_along_axis(lp, y[..., None], -1).mean()
    if not need_grad: return L
    g = np.exp(lp); np.put_along_axis(g, y[..., None], np.take_along_axis(g, y[..., None], -1) - 1, -1)
    dl = np.zeros_like(c['lg']); dl[:, :-1] = g / M
    G = {}; G['U'] = np.einsum('ntd,ntv->dv', c['h2'], dl); dh2 = dl @ p['U'].T
    G['W2'] = np.einsum('ntm,ntd->md', c['m'], dh2); dz = (dh2 @ p['W2'].T) * (c['z'] > 0)
    G['W1'] = np.einsum('ntd,ntm->dm', c['h1'], dz); dh1 = dh2 + dz @ p['W1'].T
    G['WO'] = np.einsum('nth,ntd->hd', c['o'], dh1); do = dh1 @ p['WO'].T
    da = do @ c['v'].transpose(0, 2, 1); dv = c['a'].transpose(0, 2, 1) @ do
    ds = c['a'] * (da - (da * c['a']).sum(-1, keepdims=True)) / np.sqrt(dh)
    dq = ds @ c['k']; dk = ds.transpose(0, 2, 1) @ c['q']
    G['WQ'] = np.einsum('ntd,nth->dh', c['h0'], dq); G['WK'] = np.einsum('ntd,nth->dh', c['h0'], dk)
    G['WV'] = np.einsum('ntd,nth->dh', c['h0'], dv)
    dh0 = dh1 + dq @ p['WQ'].T + dk @ p['WK'].T + dv @ p['WV'].T
    G['P'] = dh0.sum(0); G['E'] = np.zeros_like(p['E']); np.add.at(G['E'], x, dh0)
    return L, G

def charges(p):
    """Noether charges of the three continuous weight symmetries.
    QK: W_Q->W_Q A, W_K->W_K A^-T (GL(dh));  VO: W_V->W_V B, W_O->B^-1 W_O (GL(dh));
    MLP: per hidden ReLU unit j, W1[:,j]*a, W2[j,:]/a (a>0)."""
    return dict(QK=p['WQ'].T @ p['WQ'] - p['WK'].T @ p['WK'],
                VO=p['WV'].T @ p['WV'] - p['WO'] @ p['WO'].T,
                MLP=(p['W1']**2).sum(0) - (p['W2']**2).sum(1))

def charge_scale(p):
    return dict(QK=np.linalg.norm(p['WQ'].T @ p['WQ']) + np.linalg.norm(p['WK'].T @ p['WK']),
                VO=np.linalg.norm(p['WV'].T @ p['WV']) + np.linalg.norm(p['WO'] @ p['WO'].T),
                MLP=np.linalg.norm((p['W1']**2).sum(0)) + np.linalg.norm((p['W2']**2).sum(1)))

def accuracy(p, x, from_pos=1):
    pred = fwd(p, x)['lg'][:, from_pos:-1].argmax(-1); return (pred == x[:, from_pos+1:]).mean()

def train(p, x, opt, lr, steps, wd=0.0, log_every=None, on_log=None):
    m = {k: np.zeros_like(v) for k, v in p.items()}; w = {k: np.zeros_like(v) for k, v in p.items()}
    log_every = log_every or max(1, steps // 200)
    for s in range(steps + 1):
        if on_log and (s % log_every == 0 or s == steps): on_log(s, p)
        if s == steps: break
        _, G = loss_grad(p, x)
        for k in p:
            if opt == 'sgd': p[k] -= lr * G[k]
            elif opt == 'sgdm': m[k] = 0.9 * m[k] + G[k]; p[k] -= lr * m[k]
            else:
                t = s + 1; m[k] = 0.9 * m[k] + 0.1 * G[k]; w[k] = 0.999 * w[k] + 0.001 * G[k]**2
                if opt == 'adamw': p[k] *= (1 - lr * wd)
                p[k] -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(w[k] / (1 - 0.999**t)) + 1e-8)
    return p

def gradcheck():
    p = init(1); x = all_sequences()[:6]; _, G = loss_grad(p, x); r = np.random.default_rng(5); worst = 0
    for k in p:
        for _ in range(4):
            i = tuple(r.integers(0, s) for s in p[k].shape); e = 1e-6; o = p[k][i]
            p[k][i] = o + e; Lp = loss_grad(p, x, False); p[k][i] = o - e; Lm = loss_grad(p, x, False); p[k][i] = o
            fd = (Lp - Lm) / (2 * e); worst = max(worst, abs(fd - G[k][i]) / (abs(fd) + abs(G[k][i]) + 1e-12))
    return worst
