import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df, features, weight_col=None, title="Correlation Matrix", output_path=None):
    cols = features.copy()
    if weight_col and weight_col in df.columns:
        cols.append(weight_col)
    
    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, fmt=".2f")
    ax.set_title(title)
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def plot_weighted_histograms(df, features, weight_col=None, title="Feature Distributions", output_path=None):
    cols = [c for c in features if c in df.columns]
    n_cols = len(cols)
    if n_cols == 0:
        return
        
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
        
    for ax, col in zip(axes, cols):
        if weight_col and weight_col in df.columns and df[weight_col].notna().any():
            weights = df[weight_col]
        else:
            weights = None
            
        # Drop NaNs for the histogram
        valid = df[[col]].dropna()
        if weights is not None:
            valid_weights = weights.loc[valid.index]
        else:
            valid_weights = None
            
        ax.hist(valid[col], bins=30, weights=valid_weights, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        if weights is not None:
            ax.set_ylabel("Weighted Count")
        else:
            ax.set_ylabel("Count")
            
    fig.suptitle(title, fontsize=14)
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
