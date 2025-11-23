#!/usr/bin/env python3
"""
View Inter-Human Agreement Summary

Displays the cross-source summary of inter-human agreement in a readable format.
Run this after executing the analyze-agreement workflow to see per-source breakdowns.

Usage:
    python view_inter_human_summary.py
"""

import sys
from pathlib import Path
import pandas as pd


def main():
    """Display inter-human agreement summary across sources."""

    # Path to summary file
    summary_file = Path('generated/analysis/agreement/inter_human_agreement/summary_across_sources.csv')

    if not summary_file.exists():
        print(f"ERROR: Summary file not found: {summary_file}")
        print("\nPlease run the agreement analysis first:")
        print("  python main.py analyze-agreement --sources <source1> <source2> ...")
        sys.exit(1)

    # Read the summary file
    df = pd.read_csv(summary_file)

    print("=" * 100)
    print("CROSS-SOURCE INTER-HUMAN AGREEMENT SUMMARY")
    print("=" * 100)
    print("\nThis shows the average inter-human agreement for each source and dimension.")
    print("Higher Spearman/Pearson values indicate more reliable human annotations.\n")

    # Create pivot table for easy comparison
    print("\n" + "=" * 100)
    print("SPEARMAN CORRELATION BY SOURCE AND DIMENSION")
    print("=" * 100)
    pivot_spearman = df.pivot(index='dimension', columns='source', values='spearman_r')
    pivot_spearman['Overall'] = df.groupby('dimension')['overall_spearman'].first()
    print(pivot_spearman.round(4).to_string())

    print("\n" + "=" * 100)
    print("PEARSON CORRELATION BY SOURCE AND DIMENSION")
    print("=" * 100)
    pivot_pearson = df.pivot(index='dimension', columns='source', values='pearson_r')
    pivot_pearson['Overall'] = df.groupby('dimension')['overall_pearson'].first()
    print(pivot_pearson.round(4).to_string())

    print("\n" + "=" * 100)
    print("MEAN ABSOLUTE ERROR (MAE) BY SOURCE AND DIMENSION")
    print("=" * 100)
    print("(Lower is better)")
    pivot_mae = df.pivot(index='dimension', columns='source', values='mae')
    print(pivot_mae.round(4).to_string())

    print("\n" + "=" * 100)
    print("SOURCE-LEVEL SUMMARY (Average across all dimensions)")
    print("=" * 100)
    source_avg = df.groupby('source').agg({
        'spearman_r': 'mean',
        'pearson_r': 'mean',
        'mae': 'mean',
        'rmse': 'mean'
    }).round(4)
    source_avg = source_avg.sort_values('spearman_r', ascending=False)
    print(source_avg.to_string())

    # Add interpretation based on Landis & Koch
    print("\n" + "=" * 100)
    print("INTERPRETATION (Landis & Koch Scale)")
    print("=" * 100)
    for source, row in source_avg.iterrows():
        spearman = row['spearman_r']
        if spearman < 0:
            interpretation = "No agreement (negative)"
        elif spearman < 0.20:
            interpretation = "Slight agreement"
        elif spearman < 0.40:
            interpretation = "Fair agreement"
        elif spearman < 0.60:
            interpretation = "Moderate agreement"
        elif spearman < 0.80:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"

        print(f"{source:20s}: {spearman:6.4f} - {interpretation}")

    print("\n" + "=" * 100)
    print("DIMENSION-LEVEL SUMMARY (Average across all sources)")
    print("=" * 100)
    dim_avg = df.groupby('dimension').agg({
        'spearman_r': 'mean',
        'pearson_r': 'mean',
        'mae': 'mean',
        'rmse': 'mean'
    }).round(4)
    dim_avg = dim_avg.sort_values('spearman_r', ascending=False)
    print(dim_avg.to_string())

    print("\n" + "=" * 100)
    print("BEST SOURCE FOR EACH DIMENSION (Based on Spearman r)")
    print("=" * 100)
    best_per_dim = df.loc[df.groupby('dimension')['spearman_r'].idxmax()][
        ['dimension', 'source', 'spearman_r', 'n']
    ]
    print(best_per_dim.to_string(index=False))

    print("\n" + "=" * 100)
    print("FILE LOCATION")
    print("=" * 100)
    print(f"{summary_file}")
    print("=" * 100)


if __name__ == '__main__':
    main()
