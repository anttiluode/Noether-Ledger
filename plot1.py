import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('exp1.json'))
cols = {'SGD lr=1e-2': '#9ecae1', 'SGD lr=1e-1': '#4292c6', 'SGD lr=3e-1': '#08519c', 'SGD+mom lr=3e-2': '#41ab5d',
        'Adam lr=3e-3': '#e6550d', 'AdamW lr=3e-3 wd=.1': '#a63603'}
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
for k, v in r.items():
    for i, q in enumerate(['QK', 'VO', 'MLP']):
        ax[i].semilogy(v['step'], np.maximum(v[f'{q}_rel'], 1e-12), color=cols[k], label=k, lw=1.6)
    ax[3].plot(v['step'], v['acc_te'], color=cols[k], lw=1.6)
for i, t in enumerate(['Q/K charge  $W_Q^TW_Q - W_K^TW_K$', 'V/O charge  $W_V^TW_V - W_OW_O^T$', 'ReLU-unit charges  $|w_{in}|^2-|w_{out}|^2$']):
    ax[i].set_title(t + '\nrelative drift from init', fontsize=10); ax[i].set_ylim(1e-10, 1)
ax[3].set_title('held-out accuracy', fontsize=10)
for a in ax: a.set_xlabel('step'); a.grid(alpha=.3)
ax[0].legend(fontsize=8, loc='lower right'); plt.tight_layout(); plt.savefig('charges.png', dpi=110)
