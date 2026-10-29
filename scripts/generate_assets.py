import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# 1. Class Distribution
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
categories = ['No Rain (76.3%)', 'Rain (23.7%)']
counts = [5766, 1791]
colors = ['#4A90E2', '#E74C3C']
bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor='black', linewidth=0.8, alpha=0.9)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 100, f'{yval:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Target Variable Distribution (RainToday) - Melbourne Region', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Observation Count', fontsize=11, labelpad=10)
ax.set_ylim(0, 6700)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('assets/class_distribution.png')
plt.close()
print("Saved assets/class_distribution.png")

# 2. Confusion Matrices (Side-by-Side)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

cm_rf = np.array([[1094, 60], [175, 183]])
cm_lr = np.array([[1071, 83], [174, 184]])
labels = ['No Rain', 'Rain']

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=labels, yticklabels=labels, annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title('Random Forest Classifier\nAccuracy: 84.46% | Rain TPR: 51.12%', fontsize=11, fontweight='bold', pad=10)
axes[0].set_ylabel('True Label', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=labels, yticklabels=labels, annot_kws={'size': 13, 'weight': 'bold'})
axes[1].set_title('Logistic Regression Classifier\nAccuracy: 83.00% | Rain TPR: 51.40%', fontsize=11, fontweight='bold', pad=10)
axes[1].set_ylabel('True Label', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')

plt.suptitle('Test Set Confusion Matrices Comparison (n = 1,512)', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('assets/confusion_matrices.png', bbox_inches='tight')
plt.close()
print("Saved assets/confusion_matrices.png")

# 3. Feature Importance (Random Forest)
features = [
    ('Humidity 3pm', 0.1208),
    ('Pressure 3pm', 0.0908),
    ('Pressure 9am', 0.0857),
    ('Sunshine', 0.0816),
    ('Wind Gust Speed', 0.0508),
    ('Temp 3pm', 0.0457),
    ('Max Temp', 0.0414),
    ('Min Temp', 0.0404),
    ('Temp 9am', 0.0391),
    ('Humidity 9am', 0.0361),
]
features.reverse()
feat_names = [f[0] for f in features]
importances = [f[1] for f in features]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
bars = ax.barh(feat_names, importances, color='#2B5B84', edgecolor='black', linewidth=0.6, alpha=0.88)
ax.set_title('Top 10 Most Predictive Features (Random Forest)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Relative Gini Importance', fontsize=11, labelpad=10)
ax.set_xlim(0, 0.14)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.002, bar.get_y() + bar.get_height()/2.0, f'{w*100:.1f}%', ha='left', va='center', fontsize=9.5, fontweight='bold')

ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('assets/feature_importance.png')
plt.close()
print("Saved assets/feature_importance.png")

# 4. Model Comparison Metrics
metrics = ['Overall Accuracy', 'Rain Precision', 'Rain Recall (TPR)', 'Rain F1-Score']
rf_scores = [0.8446, 0.7531, 0.5112, 0.6090]
lr_scores = [0.8300, 0.6891, 0.5140, 0.5888]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
rects1 = ax.bar(x - width/2, [s*100 for s in rf_scores], width, label='Random Forest', color='#2E86AB', edgecolor='black', linewidth=0.6)
rects2 = ax.bar(x + width/2, [s*100 for s in lr_scores], width, label='Logistic Regression', color='#F6AE2D', edgecolor='black', linewidth=0.6)

ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax.set_title('Performance Benchmark: Random Forest vs. Logistic Regression', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10.5, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.6)

for rect in rects1:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('assets/model_comparison.png')
plt.close()
print("Saved assets/model_comparison.png")
