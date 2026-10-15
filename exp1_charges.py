"""Exp 1: do the Noether charges of the weight symmetries stay conserved during training?"""
import numpy as np, json, sys, model as M
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CFG = {'SGD lr=1e-2': ('sgd', 1e-2, 0), 'SGD lr=1e-1': ('sgd', 1e-1, 0), 'SGD lr=3e-1': ('sgd', 3e-1, 0),
       'SGD+mom lr=3e-2': ('sgdm', 3e-2, 0), 'Adam lr=3e-3': ('adam', 3e-3, 0), 'AdamW lr=3e-3 wd=.1': ('adamw', 3e-3, 0.1)}
only = sys.argv[2:] or list(CFG)
xtr, xte = M.split()
res = json.load(open('exp1.json')) if len(sys.argv) > 2 else {}
for name in only:
    opt, lr, wd = CFG[name]; p = M.init(); c0 = M.charges(p)
    log = {k: [] for k in ['step', 'loss', 'acc_tr', 'acc_te'] + [f'{q}_{w}' for q in c0 for w in ('drift', 'rel', 'abs')]}
    def on_log(s, p):
        c, sc = M.charges(p), M.charge_scale(p)
        log['step'].append(s); log['loss'].append(float(M.loss_grad(p, xtr, False)))
        log['acc_tr'].append(float(M.accuracy(p, xtr))); log['acc_te'].append(float(M.accuracy(p, xte)))
        for q in c:
            dr = float(np.linalg.norm(c[q] - c0[q])); log[f'{q}_drift'].append(dr)
            log[f'{q}_rel'].append(dr / sc[q]); log[f'{q}_abs'].append(float(np.linalg.norm(c[q]) / sc[q]))
    M.train(p, xtr, opt, lr, steps, wd, on_log=on_log); res[name] = log
    np.savez(f'params_{opt}_{lr}.npz', **p)
    print(f"{name:20s} loss {log['loss'][-1]:.3f} acc tr/te {log['acc_tr'][-1]:.2f}/{log['acc_te'][-1]:.2f} | " +
          ' | '.join(f"{q} rel drift {log[q+'_rel'][-1]:.1e} (|C|/scale {log[q+'_abs'][0]:.2f}->{log[q+'_abs'][-1]:.2f})" for q in c0), flush=True)
    json.dump(res, open('exp1.json', 'w'))
