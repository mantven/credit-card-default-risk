"""Aggregate evaluation figures; no individual IDs are exported."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = ['#146C94', '#DA8A2A']


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_plots(results, root):
    y, p, b = results['y_test'], results['probabilities'], results['baseline']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, prob, color in [('Logistic regression', p, COLORS[0]), ('Train prior baseline', b, COLORS[1])]:
        RocCurveDisplay.from_predictions(y, prob, name=name, ax=axes[0], color=color)
        PrecisionRecallDisplay.from_predictions(y, prob, name=name, ax=axes[1], color=color)
    axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    axes[1].axhline(y.mean(), ls=':', color='gray', label=f'Test prevalence = {y.mean():.3f}')
    axes[0].set_title('Test: ROC curve')
    axes[1].set_title('Test: precision–recall (AP = average precision)')
    axes[1].legend(fontsize=8)
    save(fig, root / 'images/roc_pr.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, color in zip(['logistic_regression', 'baseline_prior'], COLORS):
        table = results['calibration'].query('model == @name')
        nonempty = table.query('n > 0')
        score = results['metrics'].query('model == @name').iloc[0].brier
        axes[0].plot(nonempty.mean_probability, nonempty.observed_rate, 'o-', color=color, label=f'{name}, Brier={score:.4f}')
        offset = -.018 if name == 'logistic_regression' else .018
        axes[1].bar((table.bin_left + table.bin_right)/2 + offset, table.n, width=.035, color=color, alpha=.85, label=name)
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Ideal calibration')
    axes[0].set(xlabel='Mean predicted probability', ylabel='Observed default rate', title='Test: equal-width bins (10)')
    axes[1].set(xlabel='Probability bin midpoint', ylabel='Observations (count)', title=f'Bin support, test n={len(y):,}', xlim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, root / 'images/calibration.png')

    threshold = results['threshold']
    cm = confusion_matrix(y, p >= threshold)
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=17,
                    color='white' if cm[i,j] > cm.max()/2 else '#142C40')
    ax.set(xticks=[0, 1], xticklabels=['No review', 'Review'], yticks=[0, 1], yticklabels=['No default (0)', 'Default (1)'],
           xlabel='Model action', ylabel='Observed outcome', title=f'Test: fixed validation threshold {threshold:.1f}')
    save(fig, root / 'images/confusion_matrix.png')

    coef = results['coefficients'].copy()
    coef = coef.loc[coef.coefficient.abs().nlargest(18).index].sort_values('coefficient')
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(coef.feature.str.replace('categorical__', '', regex=False).str.replace('numeric__', '', regex=False),
            coef.coefficient, color=[COLORS[0] if c >= 0 else COLORS[1] for c in coef.coefficient])
    ax.axvline(0, color='gray', lw=1)
    ax.set(xlabel='Log-odds coefficient: category vs reference / numeric +1 train SD',
           title='18 largest absolute coefficients: conditional associations')
    save(fig, root / 'images/coefficients.png')
