import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- load & prep ----------
df = pd.read_csv('/Users/danielstevenrodriguezsandoval/Desktop/kedge business school/thesis/thesis_project_dsrs/data/model_comparison_results.csv').rename(
        columns={'Unnamed: 0':'model', 'Unnamed: 1':'variant'})

metrics  = ['accuracy', 'f1', 'auc']
palette  = {'baseline': '#B0B0B0', 'enhanced': '#007ACC'}
models   = df['model'].unique()
angles   = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles  += angles[:1]            # loop closure

# ---------- plot ----------
fig, axs = plt.subplots(1, len(models), subplot_kw={'polar':True},
                        figsize=(14,4))

for ax, model in zip(axs, models):
    sub = df[df['model']==model]

    base = sub[sub['variant']=='baseline'][metrics].values.flatten().tolist()
    enh  = sub[sub['variant']=='enhanced'][metrics].values.flatten().tolist()
    base += base[:1];  enh += enh[:1]

    ax.plot(angles, base, color=palette['baseline'], lw=1.5, label='Baseline')
    ax.fill(angles, base, color=palette['baseline'], alpha=.2)

    ax.plot(angles, enh,  color=palette['enhanced'], lw=1.5, label='Enhanced')
    ax.fill(angles, enh,  color=palette['enhanced'], alpha=.2)

    ax.set_title(model, size=12, pad=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_yticks([.2,.4,.6,.8,1.0])
    ax.set_yticklabels(['.2','.4','.6','.8','1.0'])
    ax.set_ylim(0,1)

axs[0].legend(loc='upper left', bbox_to_anchor=(1.1,1.15), frameon=False)
fig.suptitle("Baseline (grey) vs Enhanced (blue) – performance radar by model",
             fontsize=14, y=1.08)
plt.tight_layout()
plt.show()
