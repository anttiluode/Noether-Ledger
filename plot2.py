import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('exp2.json')); fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
cm = plt.get_cmap('coolwarm'); A = [0.25, 0.5, 1.0, 2.0, 4.0]
for i, o in enumerate(['sgd', 'adam']):
    for j, a in enumerate(A):
        v = r[f'{o} a={a}']; ax[i].plot(np.arange(len(v['curve'])) * 50, v['curve'], color=cm(j / 4), lw=2 if a == 1 else 1.4,
                                        label=f"a={a}  held-out acc {v['acc_te']:.2f}")
    ax[i].set_title(('SGD lr=0.1' if o == 'sgd' else 'Adam lr=3e-3') + ': same function at init, different gauge', fontsize=10)
    ax[i].set_xlabel('step'); ax[i].grid(alpha=.3); ax[i].legend(fontsize=8)
ax[0].set_ylabel('train loss'); plt.tight_layout(); plt.savefig('gauge_init.png', dpi=110)
