"""Render the approved composite figure from existing benchmark results."""
from pathlib import Path
import csv
import os

ROOT = Path(__file__).resolve().parents[1]
import tempfile
_mpl_cache = tempfile.TemporaryDirectory(prefix='recomb-figure-mpl-')
os.environ['MPLCONFIGDIR'] = _mpl_cache.name
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

PAPER = ROOT
METHODS = [
    ('beam', 'Beam', '#555555', 'o', '-'),
    ('sampling', 'Sampling', '#56B4E9', 's', '--'),
    ('self_refine', 'Self-Refine', '#0072B2', '^', '-.'),
    ('tot', 'ToT', '#D99B00', 'v', ':'),
    ('got', 'GoT', '#009E73', 'D', '-'),
    ('s1', 's1', '#CC79A7', 'P', '--'),
    ('voting', 'Vote', '#8C564B', 'x', '-.'),
    ('icl', 'ICL', '#9467BD', '*', ':'),
    ('reflexion', 'Reflexion', '#D55E00', 'h', '-'),
]
MODELS = [('Mistral', 'Mistral (7B)'), ('Llama', 'Llama (8B)')]

def read(name):
    with (PAPER / 'data' / name).open(newline='') as f:
        rows = list(csv.DictReader(f))
    result = {(r['model'], r['method_id']): r for r in rows}
    assert len(result) == len(rows) == 18
    return result

prefix = read('k20_prefix_macro.csv')
quality = read('table3_k20_quality.csv')
workload = read('table2_k20_workload.csv')
assert set(prefix) == set(quality) == set(workload)
for k in prefix:
    assert abs(float(prefix[k]['K20']) - float(quality[k]['SR20'])) < 1e-9

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'pdf.fonttype': 42,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

for variant in ['workload']:
    fig = plt.figure(figsize=(6.5, 3.75))
    axes = {}
    for row, bottom in [(0, .645), (1, .25)]:
        for col, left in [(0, .10), (1, .58)]:
            ax = fig.add_axes([left, bottom, .40, .265 if row == 0 else .245])
            axes[row, col] = ax
            ax.set_ylim(20, 104)
            ax.set_yticks([20, 60, 100])
            ax.grid(axis='y', color='#dddddd', linewidth=.45, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2.5, pad=2)
            if col == 0:
                ax.set_ylabel('Mean SR@K (%)' if row == 0 else 'Mean SR@20 (%)', labelpad=3)
            else:
                ax.tick_params(labelleft=False)
            letter = 'abcd'[2 * row + col]
            ax.set_title(f'({letter}) {MODELS[col][1]}', loc='left', pad=4)

    for col, (model, label) in enumerate(MODELS):
        top, lower = axes[0, col], axes[1, col]
        top.set_xlim(0, 21)
        top.set_xticks([1, 5, 10, 20])
        top.set_xlabel('Number of evaluated outputs, $K$', labelpad=2)
        for method, _, color, marker, style in METHODS:
            key = model, method
            top.plot([1, 5, 10, 20], [float(prefix[key][f'K{k}']) for k in [1, 5, 10, 20]],
                     color=color, marker=marker, linestyle=style,
                     linewidth=1.25 if method in ['got', 'reflexion'] else .85,
                     markersize=3.6, markeredgewidth=.6, zorder=3)
            x = float(workload[key]['output_tokens_mean'] if variant == 'workload' else quality[key]['Valid_slots'])
            lower.plot(x, float(prefix[key]['K20']), marker=marker, color=color,
                       linestyle='none', markersize=5.7, markeredgewidth=.7, zorder=3)
        if variant == 'workload':
            lower.set_xscale('log')
            lower.set_xlim(850, 35000)
            lower.xaxis.set_major_locator(FixedLocator([1000, 3000, 10000, 30000]))
            lower.xaxis.set_major_formatter(FixedFormatter(['1k', '3k', '10k', '30k']))
            lower.xaxis.set_minor_locator(NullLocator())
            lower.set_xlabel('Mean recorded output tokens\nper input (log scale)', labelpad=2)
        else:
            lower.set_xlim(40, 100)
            lower.set_xticks([40, 60, 80, 100])
            lower.set_xlabel('Valid outputs (%)', labelpad=2)
    handles = [Line2D([], [], color=c, marker=m, linestyle=s, linewidth=1,
                      markersize=4, markeredgewidth=.6, label=l)
               for _, l, c, m, s in METHODS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .004),
               ncol=5, frameon=False, handlelength=1.3, columnspacing=.9,
               handletextpad=.4, labelspacing=.25, borderaxespad=0)
    fig.savefig(PAPER / 'figures/prefix_coverage.pdf', metadata={'Title': 'Success rate across output prefixes and recorded generation workload'})
    fig.savefig(PAPER / 'figures/prefix_coverage.png', dpi=220)
    plt.close(fig)

_mpl_cache.cleanup()
print('Rendered Figure 1 from unchanged benchmark CSVs.')
