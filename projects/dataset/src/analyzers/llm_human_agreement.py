"""
LLM-Human Agreement Analysis

Comprehensive statistical analysis comparing LLM assessments with human annotations.

Analysis Structure:
1. Inter-Human Agreement (baseline/ground truth validation)
2. LLM-Human Agreement (model evaluation)
   - Per source, per dimension
   - Per source, overall
   - Cross-source, per dimension
   - Grand overall
3. Performance Comparison (LLM vs human baseline)
4. Summary tables (publication-ready)

Both mean and median human score aggregations analyzed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from ..processors.unified_schema import CodeSample

logger = logging.getLogger(__name__)


@dataclass
class AgreementMetrics:
    """Agreement statistics between two raters (human-human or LLM-human)."""
    rater1: str
    rater2: str
    source: str
    dimension: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    mae: float
    rmse: float
    sample_count: int
    aggregation_method: str = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'rater1': self.rater1,
            'rater2': self.rater2,
            'source': self.source,
            'dimension': self.dimension,
            'pearson_r': round(self.pearson_r, 4) if not np.isnan(self.pearson_r) else None,
            'pearson_p': round(self.pearson_p, 6) if not np.isnan(self.pearson_p) else None,
            'spearman_r': round(self.spearman_r, 4) if not np.isnan(self.spearman_r) else None,
            'spearman_p': round(self.spearman_p, 6) if not np.isnan(self.spearman_p) else None,
            'mae': round(self.mae, 4) if not np.isnan(self.mae) else None,
            'rmse': round(self.rmse, 4) if not np.isnan(self.rmse) else None,
            'n': self.sample_count
        }
        if self.aggregation_method:
            result['aggregation'] = self.aggregation_method
        return result


class LLMHumanAgreementAnalyzer:
    """
    Comprehensive LLM-human agreement analysis.

    Automatically computes:
    1. Inter-human agreement (establishes baseline)
    2. LLM-human agreement (evaluates models)
    3. Performance comparison (contextualizes results)
    """

    DIMENSIONS = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']

    def __init__(self, output_dir: str = "generated/analysis/agreement"):
        """Initialize analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set publication-quality style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'serif'

    def load_test_samples(self, source: str) -> List[CodeSample]:
        """Load test samples with human and LLM scores."""
        test_file = Path(f"generated/integrated/{source}/test.jsonl")

        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        samples = []
        with open(test_file, 'r') as f:
            for line in f:
                if line.strip():
                    sample_dict = json.loads(line)
                    sample = CodeSample.from_dict(sample_dict)
                    samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from {source}")
        return samples

    def extract_individual_human_scores(self, samples: List[CodeSample]) -> Dict[str, List[Dict]]:
        """
        Extract individual annotator scores (not aggregated).

        Returns:
            Dictionary mapping annotator_id to list of {sample_id, dimension, score}
        """
        annotator_data = defaultdict(list)

        for sample in samples:
            if not hasattr(sample, 'human_scores') or not sample.human_scores:
                continue

            for annotator_entry in sample.human_scores:
                annotator_id = annotator_entry.get('annotator_id')
                scores = annotator_entry.get('scores', {})

                for dim in self.DIMENSIONS:
                    if dim in scores and scores[dim] is not None:
                        annotator_data[annotator_id].append({
                            'sample_id': sample.submission_id,
                            'problem_id': sample.problem_id,
                            'dimension': dim,
                            'score': scores[dim]
                        })

        return dict(annotator_data)

    def compute_pairwise_agreement(self,
                                   scores1: np.ndarray,
                                   scores2: np.ndarray,
                                   rater1: str,
                                   rater2: str,
                                   source: str,
                                   dimension: str) -> AgreementMetrics:
        """Compute agreement metrics between two raters."""
        # Remove NaN pairs
        mask = ~(np.isnan(scores1) | np.isnan(scores2))
        clean1 = scores1[mask]
        clean2 = scores2[mask]

        if len(clean1) < 3:
            return AgreementMetrics(
                rater1=rater1,
                rater2=rater2,
                source=source,
                dimension=dimension,
                pearson_r=np.nan,
                pearson_p=np.nan,
                spearman_r=np.nan,
                spearman_p=np.nan,
                mae=np.nan,
                rmse=np.nan,
                sample_count=len(clean1)
            )

        # Correlations
        pearson_r, pearson_p = pearsonr(clean1, clean2)
        spearman_r, spearman_p = spearmanr(clean1, clean2)

        # Error metrics
        errors = clean2 - clean1
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        return AgreementMetrics(
            rater1=rater1,
            rater2=rater2,
            source=source,
            dimension=dimension,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            mae=mae,
            rmse=rmse,
            sample_count=len(clean1)
        )

    def analyze_inter_human_agreement(self, samples: List[CodeSample], source: str) -> pd.DataFrame:
        """
        Compute inter-human agreement (all pairwise annotator comparisons).

        Returns DataFrame with pairwise agreement metrics.
        """
        annotator_data = self.extract_individual_human_scores(samples)

        if len(annotator_data) < 2:
            logger.warning(f"Need at least 2 annotators for inter-human agreement in {source}")
            return pd.DataFrame()

        annotators = list(annotator_data.keys())
        logger.info(f"  Found {len(annotators)} annotators: {annotators}")

        results = []

        # Compute pairwise agreement for each dimension
        for dim in self.DIMENSIONS:
            # Create DataFrame for this dimension
            dim_data = {}
            for annotator_id, scores_list in annotator_data.items():
                dim_scores = {item['sample_id']: item['score']
                             for item in scores_list if item['dimension'] == dim}
                if dim_scores:
                    dim_data[annotator_id] = pd.Series(dim_scores)

            if len(dim_data) < 2:
                continue

            df = pd.DataFrame(dim_data)

            # All pairwise combinations
            for ann1, ann2 in combinations(annotators, 2):
                if ann1 not in df.columns or ann2 not in df.columns:
                    continue

                metrics = self.compute_pairwise_agreement(
                    df[ann1].values,
                    df[ann2].values,
                    ann1,
                    ann2,
                    source,
                    dim
                )
                results.append(metrics.to_dict())

        return pd.DataFrame(results)

    def aggregate_human_scores(self, sample: CodeSample, method: str) -> Optional[Dict[str, float]]:
        """Aggregate multiple human annotator scores."""
        if not hasattr(sample, 'human_scores') or not sample.human_scores:
            return None

        dimension_scores = defaultdict(list)

        for annotator_entry in sample.human_scores:
            scores = annotator_entry.get('scores', {})
            for dim in self.DIMENSIONS:
                if dim in scores and scores[dim] is not None:
                    dimension_scores[dim].append(scores[dim])

        if not dimension_scores:
            return None

        aggregated = {}
        for dim in self.DIMENSIONS:
            if dim in dimension_scores and dimension_scores[dim]:
                if method == 'mean':
                    aggregated[dim] = np.mean(dimension_scores[dim])
                elif method == 'median':
                    aggregated[dim] = np.median(dimension_scores[dim])

        return aggregated if aggregated else None

    def extract_llm_scores(self, sample: CodeSample, model_name: str) -> Optional[Dict[str, float]]:
        """Extract LLM scores for a specific model."""
        if not hasattr(sample, 'llm_scores') or not sample.llm_scores:
            return None

        for llm_entry in sample.llm_scores:
            if llm_entry.get('model_name') == model_name:
                scores = llm_entry.get('scores', {})
                return {dim: scores.get(dim) for dim in self.DIMENSIONS if dim in scores}

        return None

    def prepare_llm_human_pairs(self,
                                samples: List[CodeSample],
                                source: str,
                                human_method: str) -> Dict[str, pd.DataFrame]:
        """Prepare paired human-LLM scores for all models."""
        # Collect all unique models
        all_models = set()
        for sample in samples:
            if hasattr(sample, 'llm_scores') and sample.llm_scores:
                for llm_entry in sample.llm_scores:
                    all_models.add(llm_entry.get('model_name'))

        logger.info(f"  Found {len(all_models)} LLM models")

        # Prepare data for each model
        model_data = {}

        for model_name in all_models:
            rows = []

            for sample in samples:
                human_scores = self.aggregate_human_scores(sample, method=human_method)
                if not human_scores:
                    continue

                llm_scores = self.extract_llm_scores(sample, model_name)
                if not llm_scores:
                    continue

                row = {'sample_id': sample.submission_id, 'problem_id': sample.problem_id}

                for dim in self.DIMENSIONS:
                    if dim in human_scores and dim in llm_scores:
                        row[f'human_{dim}'] = human_scores[dim]
                        row[f'llm_{dim}'] = llm_scores[dim]

                rows.append(row)

            if rows:
                model_data[model_name] = pd.DataFrame(rows)

        return model_data

    def analyze_llm_human_per_dimension(self,
                                       model_data: Dict[str, pd.DataFrame],
                                       source: str,
                                       human_method: str) -> pd.DataFrame:
        """Analyze each dimension separately for LLM-human agreement."""
        results = []

        for model_name, df in model_data.items():
            for dim in self.DIMENSIONS:
                human_col = f'human_{dim}'
                llm_col = f'llm_{dim}'

                if human_col not in df.columns or llm_col not in df.columns:
                    continue

                metrics = self.compute_pairwise_agreement(
                    df[human_col].values,
                    df[llm_col].values,
                    'human_' + human_method,
                    model_name,
                    source,
                    dim
                )
                metrics.aggregation_method = human_method
                results.append(metrics.to_dict())

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, filepath: Path):
        """Save results to CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"  Saved: {filepath.relative_to(self.output_dir)}")

    def create_inter_human_cross_source_summary(self, all_inter_human_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-source summary of inter-human agreement.

        Aggregates pairwise agreement metrics by source and dimension,
        showing which sources have more reliable human annotations.

        Args:
            all_inter_human_df: Combined DataFrame from all sources

        Returns:
            DataFrame with per-source, per-dimension summary statistics
        """
        # Group by source and dimension, compute average agreement
        summary = all_inter_human_df.groupby(['source', 'dimension']).agg({
            'pearson_r': 'mean',
            'spearman_r': 'mean',
            'mae': 'mean',
            'rmse': 'mean',
            'n': 'sum'  # Total samples across all annotator pairs
        }).reset_index()

        # Round for readability
        summary['pearson_r'] = summary['pearson_r'].round(4)
        summary['spearman_r'] = summary['spearman_r'].round(4)
        summary['mae'] = summary['mae'].round(4)
        summary['rmse'] = summary['rmse'].round(4)

        # Pivot to make it easier to compare sources
        pivot_spearman = summary.pivot(index='dimension', columns='source', values='spearman_r')
        pivot_pearson = summary.pivot(index='dimension', columns='source', values='pearson_r')
        pivot_mae = summary.pivot(index='dimension', columns='source', values='mae')
        pivot_rmse = summary.pivot(index='dimension', columns='source', values='rmse')

        # Add overall average across sources for each dimension
        summary['overall_spearman'] = summary.groupby('dimension')['spearman_r'].transform('mean')
        summary['overall_pearson'] = summary.groupby('dimension')['pearson_r'].transform('mean')

        # Add source-level average across dimensions
        source_avg = all_inter_human_df.groupby('source').agg({
            'spearman_r': 'mean',
            'pearson_r': 'mean'
        }).round(4)

        # Store pivot tables for reference
        self._pivot_tables = {
            'spearman': pivot_spearman,
            'pearson': pivot_pearson,
            'mae': pivot_mae,
            'rmse': pivot_rmse,
            'source_average': source_avg
        }

        return summary

    def run_complete_analysis(self, sources: List[str], human_methods: Optional[List[str]] = None):
        """
        Run complete multi-dimensional agreement analysis.

        Automatically computes:
        1. Inter-human agreement (baseline)
        2. LLM-human agreement (all levels)
        3. Performance comparison
        4. Summary tables
        """
        if human_methods is None:
            human_methods = ['mean', 'median']

        logger.info(f"\n{'='*70}")
        logger.info("LLM-HUMAN AGREEMENT ANALYSIS")
        logger.info(f"{'='*70}")
        logger.info(f"Sources: {', '.join(sources)}")
        logger.info(f"Aggregation methods: {', '.join(human_methods)}")

        # STEP 1: Inter-Human Agreement (baseline)
        logger.info(f"\n{'='*70}")
        logger.info("STEP 1: INTER-HUMAN AGREEMENT (BASELINE)")
        logger.info(f"{'='*70}")

        all_inter_human = []
        for source in sources:
            logger.info(f"\nAnalyzing inter-human agreement for {source}...")
            try:
                samples = self.load_test_samples(source)
                inter_human_df = self.analyze_inter_human_agreement(samples, source)

                if not inter_human_df.empty:
                    all_inter_human.append(inter_human_df)
                    output_path = self.output_dir / "inter_human_agreement" / f"{source}_pairwise.csv"
                    self.save_results(inter_human_df, output_path)
            except Exception as e:
                logger.error(f"  Failed: {e}")

        if all_inter_human:
            combined_inter_human = pd.concat(all_inter_human, ignore_index=True)

            # Overall inter-human reliability (averaged across all sources)
            overall_reliability = combined_inter_human.groupby('dimension').agg({
                'pearson_r': 'mean',
                'spearman_r': 'mean',
                'mae': 'mean',
                'rmse': 'mean',
                'n': 'sum'
            }).reset_index()

            output_path = self.output_dir / "inter_human_agreement" / "overall_reliability.csv"
            self.save_results(overall_reliability, output_path)

            # Cross-source summary (per-source breakdown)
            cross_source_summary = self.create_inter_human_cross_source_summary(combined_inter_human)
            summary_path = self.output_dir / "inter_human_agreement" / "summary_across_sources.csv"
            self.save_results(cross_source_summary, summary_path)

            logger.info(f"\n{'='*60}")
            logger.info("INTER-HUMAN AGREEMENT SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(overall_reliability.to_string(index=False))

            logger.info(f"\n{'='*60}")
            logger.info("PER-SOURCE INTER-HUMAN AGREEMENT")
            logger.info(f"{'='*60}")
            # Display pivot table for easy comparison
            if hasattr(self, '_pivot_tables'):
                logger.info("\nSpearman Correlation by Source and Dimension:")
                logger.info(self._pivot_tables['spearman'].to_string())
                logger.info("\nSource Average (across all dimensions):")
                logger.info(self._pivot_tables['source_average'].to_string())

        # STEP 2: LLM-Human Agreement
        for human_method in human_methods:
            logger.info(f"\n{'='*70}")
            logger.info(f"STEP 2: LLM-HUMAN AGREEMENT ({human_method.upper()})")
            logger.info(f"{'='*70}")

            all_per_dim = []
            all_overall_source = []

            for source in sources:
                logger.info(f"\nProcessing {source}...")
                try:
                    samples = self.load_test_samples(source)
                    model_data = self.prepare_llm_human_pairs(samples, source, human_method)

                    if not model_data:
                        logger.warning(f"  No paired LLM-human data")
                        continue

                    # Per-dimension analysis
                    per_dim_df = self.analyze_llm_human_per_dimension(model_data, source, human_method)
                    all_per_dim.append(per_dim_df)

                    output_path = self.output_dir / "llm_human_agreement" / source / human_method / "per_dimension.csv"
                    self.save_results(per_dim_df, output_path)

                    # Overall per source
                    overall_source = per_dim_df.groupby('rater2').agg({
                        'pearson_r': 'mean',
                        'spearman_r': 'mean',
                        'mae': 'mean',
                        'rmse': 'mean',
                        'n': 'sum'
                    }).reset_index()
                    overall_source.rename(columns={'rater2': 'model'}, inplace=True)
                    overall_source['source'] = source
                    overall_source['dimension'] = 'overall'
                    all_overall_source.append(overall_source)

                    output_path = self.output_dir / "llm_human_agreement" / source / human_method / "overall_source.csv"
                    self.save_results(overall_source, output_path)

                except Exception as e:
                    logger.error(f"  Failed: {e}")

            if not all_per_dim:
                logger.warning(f"No results for {human_method}")
                continue

            combined_per_dim = pd.concat(all_per_dim, ignore_index=True)
            combined_overall_source = pd.concat(all_overall_source, ignore_index=True)

            # Cross-source per-dimension
            cross_source_per_dim = combined_per_dim.groupby(['rater2', 'dimension']).agg({
                'pearson_r': 'mean',
                'spearman_r': 'mean',
                'mae': 'mean',
                'rmse': 'mean',
                'n': 'sum'
            }).reset_index()
            cross_source_per_dim.rename(columns={'rater2': 'model'}, inplace=True)
            cross_source_per_dim = cross_source_per_dim.sort_values(['dimension', 'spearman_r'],
                                                                     ascending=[True, False])

            output_path = self.output_dir / "llm_human_agreement" / "cross_source" / human_method / "per_dimension.csv"
            self.save_results(cross_source_per_dim, output_path)

            # Grand overall
            grand_overall = combined_per_dim.groupby('rater2').agg({
                'pearson_r': 'mean',
                'spearman_r': 'mean',
                'mae': 'mean',
                'rmse': 'mean',
                'n': 'sum'
            }).reset_index()
            grand_overall.rename(columns={'rater2': 'model'}, inplace=True)
            grand_overall = grand_overall.sort_values('spearman_r', ascending=False)
            grand_overall['rank'] = range(1, len(grand_overall) + 1)
            grand_overall = grand_overall[['rank', 'model', 'pearson_r', 'spearman_r', 'mae', 'rmse', 'n']]

            output_path = self.output_dir / "llm_human_agreement" / "cross_source" / human_method / "grand_overall.csv"
            self.save_results(grand_overall, output_path)

            # Generate summaries
            self._generate_summaries(combined_overall_source, cross_source_per_dim,
                                    grand_overall, human_method)

            # Print top models
            logger.info(f"\n{'='*60}")
            logger.info(f"TOP 10 MODELS ({human_method.upper()})")
            logger.info(f"{'='*60}")
            logger.info(grand_overall.head(10).to_string(index=False))

        logger.info(f"\n{'='*70}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Results saved to: {self.output_dir}")

    def _generate_summaries(self, overall_source_df, cross_dim_df, grand_overall_df, human_method):
        """Generate summary tables."""
        summary_dir = self.output_dir / "llm_human_agreement" / "summary" / human_method
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Best model per source
        best_per_source = overall_source_df.loc[
            overall_source_df.groupby('source')['spearman_r'].idxmax()
        ][['source', 'model', 'spearman_r', 'pearson_r', 'mae', 'rmse']]
        self.save_results(best_per_source, summary_dir / "best_model_per_source.csv")

        # Best model per dimension
        best_per_dim = cross_dim_df.loc[
            cross_dim_df.groupby('dimension')['spearman_r'].idxmax()
        ][['dimension', 'model', 'spearman_r', 'pearson_r', 'mae', 'rmse']]
        self.save_results(best_per_dim, summary_dir / "best_model_per_dimension.csv")

        # Top 10 overall
        top10 = grand_overall_df.head(10)
        self.save_results(top10, summary_dir / "top_10_models.csv")
