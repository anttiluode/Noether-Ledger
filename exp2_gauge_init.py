"""Exp 2: gauge-equivalent initializations. Scale W_Q*a, W_K/a and W1*a, W2/a.
At init the network computes the SAME function (verified), but the Noether charge differs.
Question: does training remember which gauge it started in?"""
import numpy as np, json, model as M
xtr, xte = M.split(); out = {}
base = M.init(); lg0 = M.fwd(base, xtr)['lg']
for opt, lr in [('sgd', 0.1), ('adam', 3e-3)]:
    for a in [0.25, 0.5, 1.0, 2.0, 4.0]:
        p = {k: v.copy() for k, v in base.items()}
        p['WQ'] *= a; p['WK'] /= a; p['W1'] *= a; p['W2'] /= a
        same = np.abs(M.fwd(p, xtr)['lg'] - lg0).max()
        curve = []
        M.train(p, xtr, opt, lr, 1500, log_every=50, on_log=lambda s, p: curve.append(float(M.loss_grad(p, xtr, False))))
        out[f'{opt} a={a}'] = dict(curve=curve, acc_tr=float(M.accuracy(p, xtr)), acc_te=float(M.accuracy(p, xte)), init_maxdiff=float(same))
        print(f"{opt:4s} a={a:<5} init logits max diff {same:.1e} | loss {curve[0]:.3f}->{curve[-1]:.3f} at step 1500 | acc tr/te {out[f'{opt} a={a}']['acc_tr']:.2f}/{out[f'{opt} a={a}']['acc_te']:.2f}", flush=True)
json.dump(out, open('exp2.json', 'w'))
