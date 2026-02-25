"""
Analyze MADRS profile distributions:
  1. Correlation heatmap across all 10 MADRS items
  2. Swarm plots + histograms per item, split by severity band (Low / Medium / High)

Usage:
    python3 analyze_madrs_profiles.py --csv path/to/profiles_madrs.csv --output_dir path/to/output/
    python3 analyze_madrs_profiles.py --csv generated_profiles_hopkins/profiles_madrs.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set by main() from CLI args
OUTPUT_DIR = None

ITEM_COLS = [
    "REPORTED_SADNESS", "APPARENT_SADNESS", "INNER_TENSION",
    "REDUCED_SLEEP", "REDUCED_APPETITE", "CONCENTRATION_DIFFICULTIES",
    "LASSITUDE", "INABILITY_TO_FEEL", "PESSIMISTIC_THOUGHTS",
    "SUICIDAL_THOUGHTS",
]

# Short labels for readability on plots
SHORT_LABELS = {
    "REPORTED_SADNESS": "Reported\nSadness",
    "APPARENT_SADNESS": "Apparent\nSadness",
    "INNER_TENSION": "Inner\nTension",
    "REDUCED_SLEEP": "Reduced\nSleep",
    "REDUCED_APPETITE": "Reduced\nAppetite",
    "CONCENTRATION_DIFFICULTIES": "Concentration\nDifficulties",
    "LASSITUDE": "Lassitude",
    "INABILITY_TO_FEEL": "Inability\nto Feel",
    "PESSIMISTIC_THOUGHTS": "Pessimistic\nThoughts",
    "SUICIDAL_THOUGHTS": "Suicidal\nThoughts",
}

# Severity bands based on standard MADRS cutoffs (max = 60)
#   Low:    0-19   (normal / mild)
#   Medium: 20-34  (moderate)
#   High:   35-60  (severe / very severe)
BAND_BINS = [-1, 19, 34, 60]
BAND_LABELS = ["Low (0-19)", "Medium (20-34)", "High (35-60)"]
BAND_COLORS = {"Low (0-19)": "#4CAF50", "Medium (20-34)": "#FFC107", "High (35-60)": "#F44336"}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["severity_band"] = pd.cut(
        df["actual_total_score"],
        bins=BAND_BINS,
        labels=BAND_LABELS,
    )
    print(f"Loaded {len(df)} profiles")
    print(f"\nSeverity band counts:\n{df['severity_band'].value_counts().sort_index()}")
    return df


# ── 1. Correlation Heatmap ──────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    corr = df[ITEM_COLS].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
        xticklabels=[SHORT_LABELS[c] for c in ITEM_COLS],
        yticklabels=[SHORT_LABELS[c] for c in ITEM_COLS],
        ax=ax,
    )
    ax.set_title("MADRS Item Correlation Matrix (n={})".format(len(df)), fontsize=14, pad=12)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "madrs_correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 2. Per-band correlation heatmaps ────────────────────────────────────────
def plot_correlation_by_band(df):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax, band in zip(axes, BAND_LABELS):
        sub = df[df["severity_band"] == band]
        corr = sub[ITEM_COLS].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
            xticklabels=[SHORT_LABELS[c] for c in ITEM_COLS],
            yticklabels=[SHORT_LABELS[c] for c in ITEM_COLS],
            ax=ax,
        )
        ax.set_title(f"{band}  (n={len(sub)})", fontsize=12)

    fig.suptitle("MADRS Item Correlations by Severity Band", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "madrs_correlation_by_band.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 3. Swarm plots by severity band ────────────────────────────────────────


# ── 4. Histograms by severity band ─────────────────────────────────────────
def plot_histograms_by_band(df):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=False)
    axes = axes.flatten()

    bins = np.arange(-0.5, 7.5, 1)  # bin edges for integer scores 0-6

    for idx, col in enumerate(ITEM_COLS):
        ax = axes[idx]
        for band in BAND_LABELS:
            sub = df[df["severity_band"] == band][col]
            ax.hist(
                sub, bins=bins, alpha=0.55, label=band,
                color=BAND_COLORS[band], edgecolor="white", linewidth=0.5,
            )
        ax.set_title(SHORT_LABELS[col].replace("\n", " "), fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if idx % 5 == 0 else "")
        ax.set_xticks(range(7))
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("MADRS Item Score Distributions by Severity Band", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "madrs_histograms_by_band.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 5. Overall histograms per item (all scores, no band split) ──────────────
def plot_item_histograms_overall(df):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=False)
    axes = axes.flatten()

    bins = np.arange(-0.5, 7.5, 1)

    for idx, col in enumerate(ITEM_COLS):
        ax = axes[idx]
        ax.hist(
            df[col], bins=bins, color="#5C6BC0", edgecolor="white", linewidth=0.5,
        )
        ax.set_title(SHORT_LABELS[col].replace("\n", " "), fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if idx % 5 == 0 else "")
        ax.set_xticks(range(7))

    fig.suptitle("MADRS Item Score Distributions (All Profiles, n={})".format(len(df)),
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "madrs_item_histograms_overall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 6. Total score distribution ────────────────────────────────────────────
def plot_total_score_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["actual_total_score"], bins=30, edgecolor="white", color="#5C6BC0")
    for edge in [19, 34]:
        ax.axvline(edge + 0.5, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Total MADRS Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Total MADRS Scores (n={})".format(len(df)))
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "madrs_total_score_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Analyze MADRS profile distributions."
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to profiles_madrs.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots (default: profile_analysis/ next to the CSV)"
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    output_dir = args.output_dir or os.path.join(os.path.dirname(csv_path), "profile_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Make output_dir available to plotting functions via module-level global
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir

    df = load_data(csv_path)

    plot_correlation_heatmap(df)
    plot_correlation_by_band(df)
    plot_histograms_by_band(df)
    plot_item_histograms_overall(df)
    plot_total_score_distribution(df)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
