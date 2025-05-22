import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os

# Global plot style
sns.set(style="whitegrid")

# ========== Utility Functions ==========

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_and_show(fig, filepath):
    fig.tight_layout()
    fig.savefig(filepath)
    plt.show()

# ========== 1. General Distribution Plots ==========

def plot_violin_hist(df, column, title, save_dir, filename_prefix):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{title} Distribution", fontsize=18)

    # Violin + Box
    sns.violinplot(y=column, data=df, ax=axs[0], inner=None, color='skyblue')
    sns.boxplot(y=column, data=df, ax=axs[0],
                width=0.2, boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
                whiskerprops=dict(color='black', linewidth=2),
                capprops=dict(color='black', linewidth=2),
                medianprops=dict(color='red', linewidth=2))
    axs[0].set_title("Violin + Enhanced Box Plot")
    axs[0].set_ylabel(title)

    # Histogram + KDE
    sns.histplot(df[column], kde=True, ax=axs[1], color='salmon', bins=30, edgecolor='black')
    axs[1].set_title("Histogram + KDE")
    axs[1].set_xlabel(title)
    axs[1].set_ylabel("Frequency")

    save_and_show(fig, os.path.join(save_dir, f"{filename_prefix}_distribution.png"))

# ========== 2. Color-Coded Bar Plots ==========

def plot_category_bar(df, column, categorizer, labels, save_dir, filename, title):
    df['category'] = df[column].apply(categorizer)
    category_order = list(labels.keys())
    counts = df['category'].value_counts()
    counts_ordered = [counts.get(c, 0) for c in category_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(category_order, counts_ordered, color=category_order)
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    patches = [mpatches.Patch(color=color, label=labels[color]) for color in category_order]
    ax.legend(handles=patches, loc='upper right')

    save_and_show(fig, os.path.join(save_dir, filename))

# ========== 3. Color Variation Specific ==========

def plot_color_variation(df, column, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color='purple', bins=30, ax=ax)
    ax.set_title("Distribution of Color Variation Scores")
    ax.set_xlabel("Avg. RGB Std Dev")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_show(fig, os.path.join(save_dir, 'color_variation_distribution.png'))

    df['quartile'] = pd.qcut(df[column], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='quartile', y=column, data=df, palette='Oranges', ax=ax)
    ax.set_title("Color Variation by Quartile")
    ax.set_xlabel("Quartile")
    ax.set_ylabel("Color Variation Score")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_and_show(fig, os.path.join(save_dir, 'color_variation_boxplot.png'))

# ========== 4. Comparison Plot ==========

def plot_irregularity_comparison(df_all, df_melanoma, categorizer, labels, save_dir):
    for df in [df_all, df_melanoma]:
        df['category'] = df['irregularity_score'].apply(categorizer)

    category_order = list(labels.keys())
    counts_all = df_all['category'].value_counts()
    counts_melanoma = df_melanoma['category'].value_counts()
    all_ordered = [counts_all.get(c, 0) for c in category_order]
    mel_ordered = [counts_melanoma.get(c, 0) for c in category_order]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle('Irregularity Score Distribution Comparison')

    axs[0].bar(category_order, all_ordered, color=category_order)
    axs[0].set_title("All Lesions")
    axs[0].set_xlabel("Category")
    axs[0].set_ylabel("Count")
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)

    axs[1].bar(category_order, mel_ordered, color=category_order)
    axs[1].set_title("Melanoma Only")
    axs[1].set_xlabel("Category")
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    patches = [mpatches.Patch(color=c, label=labels[c]) for c in category_order]
    fig.legend(handles=patches, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))

    save_and_show(fig, os.path.join(save_dir, 'irregularity_comparison_all_vs_melanoma.png'))

# ========== Main Execution ==========

base_path = r'C:\Users\anial\Desktop\ITU\Study_Second_Semester\Projects_in_Data_Science\project\2025-FYP-Final'
result_path = os.path.join(base_path, 'result')
output_path = os.path.join(base_path, 'outputs', 'plots')

# --- Asymmetry ---
df_asym = pd.read_csv(os.path.join(result_path, 'asymmetryscores.csv')).dropna(subset=['asymmetry_score'])
save_dir = os.path.join(output_path, 'asymmetry'); ensure_dir(save_dir)
plot_violin_hist(df_asym, 'asymmetry_score', 'Asymmetry Score', save_dir, 'asymmetry')

asym_labels = {
    'green': 'Safe (<100)',
    'yellow': 'Caution (100–199)',
    'orange': 'Warning (200–299)',
    'red': 'High Risk (300+)'
}
def asym_categorizer(score):
    if score < 100: return 'green'
    elif score < 200: return 'yellow'
    elif score < 300: return 'orange'
    else: return 'red'

plot_category_bar(df_asym[df_asym['asymmetry_score'] > 0], 'asymmetry_score', asym_categorizer, asym_labels, save_dir, 'asymmetry_risk_categories.png', 'Asymmetry Risk Categories')

# --- Border Irregularity ---
df_b = pd.read_csv(os.path.join(result_path, 'borderscores.csv')).dropna(subset=['irregularity_score'])
df_b_mel = pd.read_csv(os.path.join(result_path, 'only_melanoma', 'borderscores.csv')).dropna(subset=['irregularity_score'])
save_dir = os.path.join(output_path, 'irregularity'); ensure_dir(save_dir)
plot_violin_hist(df_b, 'irregularity_score', 'Irregularity Score', save_dir, 'irregularity')

irr_labels = {
    'green': 'Low (<7.0)',
    'yellow': 'Moderate (7.0–7.99)',
    'orange': 'Irregular (8.0–10.0)',
    'red': 'Highly Irregular (>10.0)'
}
def irr_categorizer(score):
    if score < 7.0: return 'green'
    elif score < 8.0: return 'yellow'
    elif score <= 10.0: return 'orange'
    else: return 'red'

plot_irregularity_comparison(df_b, df_b_mel, irr_categorizer, irr_labels, save_dir)

# --- Color Variation ---
df_color = pd.read_csv(os.path.join(result_path, 'colorvariancescores.csv')).dropna(subset=['color_variation_score'])
save_dir = os.path.join(output_path, 'color_variation'); ensure_dir(save_dir)
plot_color_variation(df_color, 'color_variation_score', save_dir)

# --- Blue-White Veil ---
df_veil = pd.read_csv(os.path.join(result_path, 'blue_white_veil.csv'))
df_veil = df_veil[df_veil['blue_white_veil_score'] != 'N/A'].copy()
df_veil['blue_white_veil_score'] = df_veil['blue_white_veil_score'].astype(float)
save_dir = os.path.join(output_path, 'veil'); ensure_dir(save_dir)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_veil['blue_white_veil_score'], kde=True, color='dodgerblue', bins=30, ax=ax)
ax.set_title("Distribution of Blue-White Veil Scores")
ax.set_xlabel("Veil Score (%)")
ax.set_ylabel("Frequency")
ax.grid(axis='y', linestyle='--', alpha=0.7)
save_and_show(fig, os.path.join(save_dir, 'veil_score_distribution.png'))

def veil_bin(score):
    if score < 5: return 'Very Low'
    elif score < 10: return 'Low'
    elif score < 20: return 'Moderate'
    else: return 'High'

df_veil['veil_category'] = df_veil['blue_white_veil_score'].apply(veil_bin)
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='veil_category', data=df_veil, order=['Very Low', 'Low', 'Moderate', 'High'], palette='Blues', ax=ax)
ax.set_title("Blue-White Veil Score Categories")
ax.set_xlabel("Veil Risk Category")
ax.set_ylabel("Number of Lesions")
ax.grid(axis='y', linestyle='--', alpha=0.7)
save_and_show(fig, os.path.join(save_dir, 'veil_score_categories.png'))
