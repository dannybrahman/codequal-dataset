#!/usr/bin/env python3
"""
View LLM-Human Agreement Results

Displays LLM-human agreement analysis results, showing which models best match
human intuition across different sources and dimensions.

Usage:
    python view_llm_human_agreement.py [--method mean|median]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd


def display_top_models(method='mean'):
    """Display top 10 models overall."""
    file_path = Path(f'generated/analysis/agreement/llm_human_agreement/summary/{method}/top_10_models.csv')

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    df = pd.read_csv(file_path)

    print("\n" + "=" * 100)
    print(f"TOP 10 MODELS - OVERALL ({method.upper()} aggregation)")
    print("=" * 100)
    print("\nModels ranked by Spearman correlation with human annotations")
    print("(Higher correlation = better match with human intuition)\n")
    print(df.to_string(index=False))

    return True


def display_best_per_source(method='mean'):
    """Display best model for each source."""
    file_path = Path(f'generated/analysis/agreement/llm_human_agreement/summary/{method}/best_model_per_source.csv')

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    df = pd.read_csv(file_path)

    print("\n" + "=" * 100)
    print(f"BEST MODEL PER SOURCE ({method.upper()} aggregation)")
    print("=" * 100)
    print("\nWhich model performs best on each dataset?\n")
    print(df.to_string(index=False))

    return True


def display_best_per_dimension(method='mean'):
    """Display best model for each dimension."""
    file_path = Path(f'generated/analysis/agreement/llm_human_agreement/summary/{method}/best_model_per_dimension.csv')

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    df = pd.read_csv(file_path)

    print("\n" + "=" * 100)
    print(f"BEST MODEL PER DIMENSION ({method.upper()} aggregation)")
    print("=" * 100)
    print("\nWhich model performs best on each quality dimension?\n")
    print(df.to_string(index=False))

    return True


def display_source_breakdown(method='mean'):
    """Display per-source model performance breakdown."""
    sources = ['codeeval', 'codesearchnet', 'humaneval-x', 'mbpp', 'codenet']
    available_sources = []

    print("\n" + "=" * 100)
    print(f"PER-SOURCE MODEL RANKINGS ({method.upper()} aggregation)")
    print("=" * 100)

    for source in sources:
        file_path = Path(f'generated/analysis/agreement/llm_human_agreement/{source}/{method}/overall_source.csv')

        if not file_path.exists():
            continue

        available_sources.append(source)
        df = pd.read_csv(file_path)

        # Sort by spearman_r descending
        df = df.sort_values('spearman_r', ascending=False).head(10)

        print(f"\n{'-' * 100}")
        print(f"SOURCE: {source.upper()}")
        print(f"{'-' * 100}")
        print(df.to_string(index=False))

    if not available_sources:
        print("No source data found.")
        return False

    print(f"\n\nAnalyzed {len(available_sources)} sources: {', '.join(available_sources)}")
    return True


def display_grand_overall(method='mean'):
    """Display grand overall statistics."""
    file_path = Path(f'generated/analysis/agreement/llm_human_agreement/cross_source/{method}/grand_overall.csv')

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    df = pd.read_csv(file_path)

    # Sort by spearman_r descending
    df = df.sort_values('spearman_r', ascending=False)

    print("\n" + "=" * 100)
    print(f"GRAND OVERALL - ALL SOURCES × ALL DIMENSIONS ({method.upper()} aggregation)")
    print("=" * 100)
    print("\nComprehensive ranking across all datasets and quality dimensions")
    print("(This is the most holistic view of model performance)\n")
    print(df.head(15).to_string(index=False))

    return True


def display_comparison_with_baseline(method='mean'):
    """Compare LLM performance with inter-human agreement baseline."""
    # Load inter-human agreement
    inter_human_path = Path('generated/analysis/agreement/inter_human_agreement/overall_reliability.csv')
    if not inter_human_path.exists():
        print("Note: Inter-human baseline not available for comparison")
        return False

    inter_human = pd.read_csv(inter_human_path)

    # Load top model performance
    top_models_path = Path(f'generated/analysis/agreement/llm_human_agreement/summary/{method}/top_10_models.csv')
    if not top_models_path.exists():
        return False

    top_models = pd.read_csv(top_models_path)
    best_model = top_models.iloc[0]

    print("\n" + "=" * 100)
    print(f"CONTEXT: LLM vs INTER-HUMAN AGREEMENT BASELINE ({method.upper()})")
    print("=" * 100)
    print("\nHow do LLMs compare to human annotator agreement?")
    print("\nInter-Human Agreement (Baseline):")
    print(f"  Average Spearman r: {inter_human['spearman_r'].mean():.4f}")
    print(f"  Range: {inter_human['spearman_r'].min():.4f} - {inter_human['spearman_r'].max():.4f}")

    print(f"\nBest LLM ({best_model['model']}):")
    print(f"  Spearman r: {best_model['spearman_r']:.4f}")
    print(f"  Sample size: {int(best_model['n'])} assessments")

    # Interpretation
    ratio = best_model['spearman_r'] / inter_human['spearman_r'].mean()
    print(f"\nBest LLM achieves {ratio:.1%} of inter-human agreement")

    if ratio > 1.0:
        print("⚠️  Note: LLM appears to exceed inter-human agreement. This may indicate:")
        print("   - LLM is very consistent across assessments")
        print("   - Sample size differences between inter-human and LLM-human")
        print("   - Different sources have different baseline agreements")
    elif ratio > 0.8:
        print("✓ Excellent: LLM approaches human-level agreement")
    elif ratio > 0.6:
        print("✓ Good: LLM shows substantial agreement with humans")
    elif ratio > 0.4:
        print("⚠️  Moderate: LLM has room for improvement")
    else:
        print("⚠️  Weak: Significant gap between LLM and human agreement")

    return True


def compare_mean_vs_median():
    """
    Compare mean vs median aggregation methods to determine which is better.

    Analyzes:
    1. Top model performance under each method
    2. Consistency of rankings across methods
    3. Error metrics (MAE, RMSE) comparison
    4. Recommendation based on empirical data
    """
    print("\n" + "=" * 100)
    print("MEAN vs MEDIAN AGGREGATION COMPARISON")
    print("=" * 100)
    print("\nComparing human score aggregation methods to determine which is better...\n")

    # Check if both methods exist
    mean_path = Path('generated/analysis/agreement/llm_human_agreement/summary/mean/top_10_models.csv')
    median_path = Path('generated/analysis/agreement/llm_human_agreement/summary/median/top_10_models.csv')

    if not mean_path.exists() or not median_path.exists():
        print("ERROR: Both mean and median results are required for comparison")
        print("\nPlease run the agreement analysis with both methods:")
        print("  python main.py analyze-agreement --sources <sources> --human-methods mean median")
        return False

    mean_results = pd.read_csv(mean_path)
    median_results = pd.read_csv(median_path)

    # 1. Compare top model performance
    print("=" * 100)
    print("1. TOP MODEL PERFORMANCE")
    print("=" * 100)

    mean_best = mean_results.iloc[0]
    median_best = median_results.iloc[0]

    print(f"\nBest model with MEAN aggregation:")
    print(f"  Model: {mean_best['model']}")
    print(f"  Spearman r: {mean_best['spearman_r']:.4f}")
    print(f"  Pearson r: {mean_best['pearson_r']:.4f}")
    print(f"  MAE: {mean_best['mae']:.4f}")
    print(f"  RMSE: {mean_best['rmse']:.4f}")

    print(f"\nBest model with MEDIAN aggregation:")
    print(f"  Model: {median_best['model']}")
    print(f"  Spearman r: {median_best['spearman_r']:.4f}")
    print(f"  Pearson r: {median_best['pearson_r']:.4f}")
    print(f"  MAE: {median_best['mae']:.4f}")
    print(f"  RMSE: {median_best['rmse']:.4f}")

    # 2. Compare average performance across top 10
    print("\n" + "=" * 100)
    print("2. AVERAGE PERFORMANCE (TOP 10 MODELS)")
    print("=" * 100)

    mean_avg = mean_results.head(10).agg({
        'spearman_r': 'mean',
        'pearson_r': 'mean',
        'mae': 'mean',
        'rmse': 'mean'
    })

    median_avg = median_results.head(10).agg({
        'spearman_r': 'mean',
        'pearson_r': 'mean',
        'mae': 'mean',
        'rmse': 'mean'
    })

    comparison_df = pd.DataFrame({
        'MEAN': mean_avg,
        'MEDIAN': median_avg,
        'Difference': mean_avg - median_avg
    })

    print("\n" + comparison_df.round(4).to_string())

    # 3. Ranking consistency
    print("\n" + "=" * 100)
    print("3. RANKING CONSISTENCY")
    print("=" * 100)

    mean_top10_models = set(mean_results.head(10)['model'])
    median_top10_models = set(median_results.head(10)['model'])

    overlap = mean_top10_models & median_top10_models
    overlap_pct = len(overlap) / 10 * 100

    print(f"\nTop 10 models appearing in both methods: {len(overlap)}/10 ({overlap_pct:.0f}%)")
    print(f"Models only in MEAN top 10: {mean_top10_models - median_top10_models}")
    print(f"Models only in MEDIAN top 10: {median_top10_models - mean_top10_models}")

    # Check if same model is #1 in both
    same_winner = mean_best['model'] == median_best['model']
    if same_winner:
        print(f"\n✓ Same best model in both methods: {mean_best['model']}")
    else:
        print(f"\n⚠️  Different best models:")
        print(f"  MEAN: {mean_best['model']}")
        print(f"  MEDIAN: {median_best['model']}")

    # 4. Recommendation
    print("\n" + "=" * 100)
    print("4. RECOMMENDATION")
    print("=" * 100)

    # Load inter-human agreement for context
    inter_human_path = Path('generated/analysis/agreement/inter_human_agreement/summary_across_sources.csv')
    inter_human_agreement = None
    if inter_human_path.exists():
        inter_human_df = pd.read_csv(inter_human_path)
        inter_human_agreement = inter_human_df['spearman_r'].mean()

    # Determine winner based on multiple criteria (weighted by importance)
    criteria_scores = {'mean': 0.0, 'median': 0.0}
    criteria_weights = {}

    # Criterion 1: Best model performance (WEIGHT: 3.0 - most important!)
    # The top-performing model is what you'll actually use
    mean_best_spearman = mean_best['spearman_r']
    median_best_spearman = median_best['spearman_r']

    if median_best_spearman > mean_best_spearman:
        criteria_scores['median'] += 3.0
        best_model_winner = 'MEDIAN'
        best_model_msg = f"Best model ({median_best['model']}) achieves higher correlation with MEDIAN"
    elif mean_best_spearman > median_best_spearman:
        criteria_scores['mean'] += 3.0
        best_model_winner = 'MEAN'
        best_model_msg = f"Best model ({mean_best['model']}) achieves higher correlation with MEAN"
    else:
        best_model_winner = 'TIE'
        best_model_msg = "Best models perform equally"
    criteria_weights['Best model performance'] = 3.0

    # Criterion 2: Average top-10 performance (WEIGHT: 2.0)
    if mean_avg['spearman_r'] > median_avg['spearman_r']:
        criteria_scores['mean'] += 2.0
        avg_winner = 'MEAN'
    else:
        criteria_scores['median'] += 2.0
        avg_winner = 'MEDIAN'
    criteria_weights['Average top-10 performance'] = 2.0

    # Criterion 3: Inter-human agreement context (WEIGHT: 2.0)
    # If inter-human agreement is low/moderate, median is more robust
    if inter_human_agreement is not None:
        if inter_human_agreement < 0.4:  # Fair to moderate agreement
            criteria_scores['median'] += 2.0
            inter_human_winner = 'MEDIAN'
            inter_human_msg = f"Inter-human agreement is moderate ({inter_human_agreement:.3f}) - MEDIAN more robust"
        elif inter_human_agreement > 0.6:  # Substantial agreement
            criteria_scores['mean'] += 2.0
            inter_human_winner = 'MEAN'
            inter_human_msg = f"Inter-human agreement is high ({inter_human_agreement:.3f}) - MEAN appropriate"
        else:  # Moderate agreement
            criteria_scores['median'] += 1.0
            criteria_scores['mean'] += 1.0
            inter_human_winner = 'BOTH'
            inter_human_msg = f"Inter-human agreement is moderate ({inter_human_agreement:.3f}) - slight preference for MEDIAN"
        criteria_weights['Inter-human agreement'] = 2.0
    else:
        inter_human_winner = 'UNKNOWN'
        inter_human_msg = "Inter-human agreement data not available"
        criteria_weights['Inter-human agreement'] = 0.0

    # Criterion 4: Error metrics (WEIGHT: 1.0)
    if mean_avg['mae'] < median_avg['mae']:
        criteria_scores['mean'] += 1.0
        error_winner = 'MEAN'
    else:
        criteria_scores['median'] += 1.0
        error_winner = 'MEDIAN'
    criteria_weights['Lower error (MAE)'] = 1.0

    # Criterion 5: Ranking consistency
    if overlap_pct >= 80:
        consistency_msg = "High consistency (≥80%) - either method is acceptable"
    elif overlap_pct >= 60:
        consistency_msg = "Moderate consistency (60-80%) - prefer method with better metrics"
    else:
        consistency_msg = "Low consistency (<60%) - methods produce different results, choose carefully"

    print(f"\nWeighted Criteria Analysis:")
    print(f"  [Weight: {criteria_weights['Best model performance']:.1f}] Best model: {best_model_winner} - {best_model_msg}")
    print(f"     MEAN: {mean_best['model']} (Spearman {mean_best_spearman:.4f})")
    print(f"     MEDIAN: {median_best['model']} (Spearman {median_best_spearman:.4f})")
    print(f"\n  [Weight: {criteria_weights['Average top-10 performance']:.1f}] Average top-10 Spearman: {avg_winner}")
    print(f"     MEAN: {mean_avg['spearman_r']:.4f}, MEDIAN: {median_avg['spearman_r']:.4f}")
    print(f"\n  [Weight: {criteria_weights['Inter-human agreement']:.1f}] {inter_human_msg}")
    print(f"\n  [Weight: {criteria_weights['Lower error (MAE)']:.1f}] Lower MAE: {error_winner}")
    print(f"     MEAN: {mean_avg['mae']:.4f}, MEDIAN: {median_avg['mae']:.4f}")
    print(f"\n  Ranking consistency: {consistency_msg}")

    # Final recommendation
    total_weight = sum(criteria_weights.values())
    print("\n" + "-" * 100)

    if criteria_scores['median'] > criteria_scores['mean']:
        recommended = 'MEDIAN'
        confidence = criteria_scores['median'] / total_weight * 100
    elif criteria_scores['mean'] > criteria_scores['median']:
        recommended = 'MEAN'
        confidence = criteria_scores['mean'] / total_weight * 100
    else:
        recommended = 'EITHER (tie)'
        confidence = 50.0

    print(f"\n🎯 RECOMMENDED AGGREGATION METHOD: {recommended}")
    print(f"   Weighted score: MEAN={criteria_scores['mean']:.1f}, MEDIAN={criteria_scores['median']:.1f} (out of {total_weight:.1f})")
    print(f"   Confidence: {confidence:.0f}%")

    # Contextual advice
    print("\n" + "-" * 100)
    print("\nContextual Considerations:")
    print("  • MEAN: Better for normally distributed scores, more sensitive to all annotators")
    print("  • MEDIAN: More robust to outliers, better when annotators have different strictness")
    print("  • If results are similar, MEAN is standard practice in ML literature")
    print("  • Consider reporting both methods in your paper to show robustness")

    if abs(mean_avg['spearman_r'] - median_avg['spearman_r']) < 0.02:
        print("\n  ✓ Results are very similar (Spearman diff < 0.02) - either method is valid")
    elif abs(mean_avg['spearman_r'] - median_avg['spearman_r']) < 0.05:
        print("\n  ✓ Results are reasonably similar (Spearman diff < 0.05) - choose based on preference")
    else:
        print("\n  ⚠️  Substantial difference in results - investigate human score distributions")

    return True


def main():
    """View LLM-human agreement results - auto-detects available aggregation methods."""

    # Check which aggregation methods have results
    base_dir = Path('generated/analysis/agreement/llm_human_agreement/summary')

    if not base_dir.exists():
        print("ERROR: No LLM-human agreement results found")
        print("\nPlease run the agreement analysis first:")
        print("  python main.py analyze-agreement --sources <source1> <source2> ...")
        sys.exit(1)

    mean_dir = base_dir / 'mean'
    median_dir = base_dir / 'median'

    has_mean = mean_dir.exists() and (mean_dir / 'top_10_models.csv').exists()
    has_median = median_dir.exists() and (median_dir / 'top_10_models.csv').exists()

    if not has_mean and not has_median:
        print("ERROR: No complete results found for mean or median aggregation")
        print("\nPlease run the agreement analysis first:")
        print("  python main.py analyze-agreement --sources <source1> <source2> ...")
        sys.exit(1)

    print("=" * 100)
    print("LLM-HUMAN AGREEMENT ANALYSIS RESULTS")
    print("=" * 100)
    print(f"\nResults directory: generated/analysis/agreement/llm_human_agreement/")
    print(f"Available aggregation methods: {', '.join([m for m in ['mean', 'median'] if (m == 'mean' and has_mean) or (m == 'median' and has_median)])}")

    # Display results for each available method
    for method in ['mean', 'median']:
        if (method == 'mean' and not has_mean) or (method == 'median' and not has_median):
            continue

        print("\n" + "=" * 100)
        print(f"RESULTS WITH {method.upper()} AGGREGATION")
        print("=" * 100)

        success = True
        success &= display_top_models(method)
        success &= display_best_per_source(method)
        success &= display_best_per_dimension(method)
        success &= display_grand_overall(method)
        success &= display_source_breakdown(method)
        display_comparison_with_baseline(method)

        if not success:
            print(f"\n⚠️  Some sections could not be displayed for {method}")

    # If both methods available, show comparison
    if has_mean and has_median:
        print("\n" + "=" * 100)
        print("BONUS: AUTOMATIC COMPARISON")
        print("=" * 100)
        compare_mean_vs_median()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("""
Key output files:
  • llm_human_agreement/summary/{method}/top_10_models.csv
  • llm_human_agreement/summary/{method}/best_model_per_source.csv
  • llm_human_agreement/summary/{method}/best_model_per_dimension.csv
  • llm_human_agreement/cross_source/{method}/grand_overall.csv
  • llm_human_agreement/{source}/{method}/overall_source.csv

This viewer automatically displays all available aggregation methods.
""")


if __name__ == '__main__':
    main()
