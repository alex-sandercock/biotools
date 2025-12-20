#!/usr/bin/env python3
"""
Generate educational PCA example figures for the BIGapp workshop.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = "/Users/aherranssanderco/Documents/GitHub/biotools/images"

# Common plot settings
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14


def create_pca_example():
    """Basic PCA scatter plot - single cloud of points."""
    # Generate a single cloud of points
    n_points = 150
    pc1 = np.random.normal(0, 2, n_points)
    pc2 = np.random.normal(0, 1.5, n_points)

    fig, ax = plt.subplots()
    ax.scatter(pc1, pc2, c='#666666', alpha=0.7, s=50, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('PC1 (45.2%)')
    ax.set_ylabel('PC2 (18.7%)')
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6, 6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pca-example.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created pca-example.png")


def create_pca_clusters():
    """PCA with distinct clusters - 3 well-separated groups."""
    n_per_group = 60

    # Group 1 - bottom left
    g1_pc1 = np.random.normal(-4, 0.8, n_per_group)
    g1_pc2 = np.random.normal(-2, 0.7, n_per_group)

    # Group 2 - top
    g2_pc1 = np.random.normal(0, 0.9, n_per_group)
    g2_pc2 = np.random.normal(3, 0.8, n_per_group)

    # Group 3 - right
    g3_pc1 = np.random.normal(4, 0.8, n_per_group)
    g3_pc2 = np.random.normal(-1, 0.7, n_per_group)

    fig, ax = plt.subplots()
    ax.scatter(g1_pc1, g1_pc2, c='#E74C3C', alpha=0.7, s=50, label='Population A', edgecolors='white', linewidths=0.5)
    ax.scatter(g2_pc1, g2_pc2, c='#3498DB', alpha=0.7, s=50, label='Population B', edgecolors='white', linewidths=0.5)
    ax.scatter(g3_pc1, g3_pc2, c='#2ECC71', alpha=0.7, s=50, label='Population C', edgecolors='white', linewidths=0.5)
    ax.set_xlabel('PC1 (52.1%)')
    ax.set_ylabel('PC2 (24.3%)')
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6, 6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pca-clusters.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created pca-clusters.png")


def create_pca_continuous():
    """PCA with continuous variation - gradient/cline pattern."""
    n_points = 200

    # Create a gradient along a diagonal
    t = np.linspace(0, 1, n_points)
    pc1 = t * 10 - 5 + np.random.normal(0, 0.6, n_points)
    pc2 = t * 6 - 3 + np.random.normal(0, 0.5, n_points)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pc1, pc2, c=t, cmap='viridis', alpha=0.7, s=50, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('PC1 (38.9%)')
    ax.set_ylabel('PC2 (21.4%)')
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    cbar = plt.colorbar(scatter, ax=ax, label='Geographic gradient')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6, 6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pca-continuous.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created pca-continuous.png")


def create_pca_outliers():
    """PCA with clear outliers - main cluster with distant points."""
    # Main cluster
    n_main = 140
    main_pc1 = np.random.normal(0, 1.5, n_main)
    main_pc2 = np.random.normal(0, 1.2, n_main)

    # Outliers
    outlier_pc1 = np.array([-6, 5.5, 6])
    outlier_pc2 = np.array([4, 4.5, -4])

    fig, ax = plt.subplots()
    ax.scatter(main_pc1, main_pc2, c='#666666', alpha=0.7, s=50, label='Main samples', edgecolors='white', linewidths=0.5)
    ax.scatter(outlier_pc1, outlier_pc2, c='#E74C3C', alpha=0.9, s=100, label='Outliers', edgecolors='darkred', linewidths=1.5, marker='o')
    ax.set_xlabel('PC1 (42.8%)')
    ax.set_ylabel('PC2 (19.1%)')
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6, 6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pca-outliers.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created pca-outliers.png")


if __name__ == "__main__":
    print("Generating PCA example figures...")
    create_pca_example()
    create_pca_clusters()
    create_pca_continuous()
    create_pca_outliers()
    print("Done!")
