"""Generate Q34 Platonic space projections as PNG images."""
import json, numpy as np

with open('THOUGHT/LAB/FORMULA/v2_2/q34_platonic/platonic_3d.json') as f:
    data = json.load(f)

words = data['words']
anchors = set(data['anchors'])
ml = np.array([[p['x'], p['y'], p['z']] for p in data['models']['MiniLM']])
mp = np.array([[p['x'], p['y'], p['z']] for p in data['models']['MPNet']])
is_anchor = np.array([w in anchors for w in words])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(21, 14))

# Row 1: MiniLM projections
# Row 2: MPNet projections
views = [((0, 1), 'XY (front)'), ((0, 2), 'XZ (top)'), ((1, 2), 'YZ (side)')]

for col, ((dx, dy), title) in enumerate(views):
    for row, (label, pts, color) in enumerate([('MiniLM', ml, '#44aaff'), ('MPNet', mp, '#ff44aa')]):
        ax = axes[row, col]
        ax.scatter(pts[is_anchor, dx], pts[is_anchor, dy], c=color, s=80, alpha=0.9, zorder=3, label='Anchor')
        ax.scatter(pts[~is_anchor, dx], pts[~is_anchor, dy], c=color, s=35, alpha=0.35, zorder=2, label='Word')
        for i in range(len(words)):
            if words[i] in anchors:
                ax.annotate(words[i], (pts[i, dx], pts[i, dy]), fontsize=6, alpha=0.7, color=color)
        ax.set_title(f'{label} — {title}', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

# Col 3: Edge comparison bar chart
ax = axes[0, 2]
edge_diffs = [abs(e['dist_ml'] - e['dist_mp']) for e in data['edges']]
edge_labels = [f"{e['a']}-{e['b']}" for e in data['edges']]
median_diff = np.median(edge_diffs)
colors = ['#44ff66' if d < median_diff else '#ff6644' for d in edge_diffs]
ax.barh(range(len(edge_diffs)), edge_diffs, color=colors)
ax.set_yticks(range(len(edge_diffs)))
ax.set_yticklabels(edge_labels, fontsize=7)
ax.set_xlabel('Distance difference |MiniLM - MPNet|')
ax.set_title('Opposite Pair Convergence\n(green = models agree, red = diverge)', fontsize=10)
ax.invert_yaxis()

# Row 2, Col 3: Summary
ax2 = axes[1, 2]
ax2.axis('off')
summary = f"""PLATONIC SPACE Q34

106 words × 2 models
PCA-3D + Procrustes on STABLE_32

Anchors (STABLE_32): {len(anchors)}
Opposite pairs: {len(data['edges'])}

Mean edge distance diff: {np.mean(edge_diffs):.2f}
Platonic agreement: {sum(1 for d in edge_diffs if d < median_diff)}/{len(edge_diffs)}

Blue = MiniLM (384d)
Pink = MPNet (768d)
Large = Anchor word
Small = Extended word
"""
ax2.text(0.1, 0.5, summary, fontsize=12, fontfamily='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('THOUGHT/LAB/FORMULA/v2_2/q34_platonic/platonic_views.png', dpi=150, bbox_inches='tight')
print('Saved platonic_views.png')
