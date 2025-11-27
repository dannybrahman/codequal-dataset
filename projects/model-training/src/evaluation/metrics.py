"""
Evaluation Metrics

Same metrics as dataset project's agreement analysis.
"""

from typing import Dict, List
import numpy as np
from scipy.stats import spearmanr, pearsonr


def compute_agreement_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    dimension_names: List[str] = None
) -> Dict:
    """
    Compute agreement metrics between predictions and targets.

    Uses same metrics as LLM-human agreement analysis:
    - Spearman correlation
    - Pearson correlation
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)

    Args:
        predictions: Predicted scores [n_samples, n_dimensions]
        targets: Ground truth scores [n_samples, n_dimensions]
        dimension_names: Optional names for dimensions

    Returns:
        Dict with overall and per-dimension metrics
    """
    if dimension_names is None:
        dimension_names = ['functionality', 'readability', 'idiomatic',
                          'error_handling', 'efficiency']

    n_samples, n_dimensions = predictions.shape
    assert predictions.shape == targets.shape, "Shape mismatch"

    metrics = {
        'n_samples': n_samples,
        'overall': {},
        'per_dimension': {}
    }

    # Overall metrics (averaged across all dimensions)
    all_preds = predictions.flatten()
    all_targets = targets.flatten()

    metrics['overall'] = _compute_metrics_for_pair(all_preds, all_targets)

    # Per-dimension metrics
    for i, dim_name in enumerate(dimension_names):
        dim_preds = predictions[:, i]
        dim_targets = targets[:, i]

        metrics['per_dimension'][dim_name] = _compute_metrics_for_pair(
            dim_preds, dim_targets
        )

    return metrics


def _compute_metrics_for_pair(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Compute metrics for a single prediction-target pair."""
    # Remove any NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    preds_clean = predictions[mask]
    targets_clean = targets[mask]

    if len(preds_clean) == 0:
        return {
            'spearman_r': np.nan,
            'spearman_p': np.nan,
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'n': 0
        }

    # Correlation metrics
    try:
        spearman_r, spearman_p = spearmanr(preds_clean, targets_clean)
    except:
        spearman_r, spearman_p = np.nan, np.nan

    try:
        pearson_r, pearson_p = pearsonr(preds_clean, targets_clean)
    except:
        pearson_r, pearson_p = np.nan, np.nan

    # Error metrics
    mae = np.mean(np.abs(preds_clean - targets_clean))
    rmse = np.sqrt(np.mean((preds_clean - targets_clean) ** 2))

    return {
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'mae': float(mae),
        'rmse': float(rmse),
        'n': int(len(preds_clean))
    }


def format_metrics_table(metrics: Dict) -> str:
    """Format metrics as a readable table."""
    lines = []

    lines.append("=" * 80)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 80)
    lines.append(f"Total samples: {metrics['n_samples']}")
    lines.append("")

    # Overall metrics
    lines.append("OVERALL (All Dimensions)")
    lines.append("-" * 80)
    overall = metrics['overall']
    lines.append(f"  Spearman r: {overall['spearman_r']:.4f} (p={overall['spearman_p']:.4e})")
    lines.append(f"  Pearson r:  {overall['pearson_r']:.4f} (p={overall['pearson_p']:.4e})")
    lines.append(f"  MAE:        {overall['mae']:.4f}")
    lines.append(f"  RMSE:       {overall['rmse']:.4f}")
    lines.append("")

    # Per-dimension metrics
    lines.append("PER-DIMENSION")
    lines.append("-" * 80)
    for dim_name, dim_metrics in metrics['per_dimension'].items():
        lines.append(f"\n{dim_name.upper()}")
        lines.append(f"  Spearman r: {dim_metrics['spearman_r']:.4f}")
        lines.append(f"  Pearson r:  {dim_metrics['pearson_r']:.4f}")
        lines.append(f"  MAE:        {dim_metrics['mae']:.4f}")
        lines.append(f"  RMSE:       {dim_metrics['rmse']:.4f}")

    # Per-source metrics (if available)
    if 'per_source' in metrics:
        lines.append("")
        lines.append("=" * 80)
        lines.append("PER-SOURCE BREAKDOWN")
        lines.append("=" * 80)

        for source, source_metrics in sorted(metrics['per_source'].items()):
            lines.append("")
            lines.append(f"{source.upper()} (n={source_metrics['n_samples']})")
            lines.append("-" * 40)

            # Overall for this source
            src_overall = source_metrics['overall']
            lines.append(f"  Overall Spearman r: {src_overall['spearman_r']:.4f}")
            lines.append(f"  Overall MAE:        {src_overall['mae']:.4f}")

            # Per-dimension for this source
            lines.append("  Per-dimension Spearman r:")
            for dim_name, dim_metrics in source_metrics['per_dimension'].items():
                lines.append(f"    {dim_name}: {dim_metrics['spearman_r']:.4f}")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
