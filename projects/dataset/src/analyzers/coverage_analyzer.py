#!/usr/bin/env python3
"""
Coverage Analyzer - Analyze coverage of human and LLM scores across datasets.

This analyzer provides a comprehensive view of:
- Which models have scored which samples
- Coverage percentage per model per dataset
- Human annotation coverage

Usage:
    python main.py analyze-coverage --split test
    python main.py analyze-coverage --split test --sources codeeval mbpp
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Analyze coverage of human and LLM scores across datasets."""

    def __init__(self):
        """Initialize the coverage analyzer."""
        self.integrated_path = Path("generated") / "integrated"

    def get_available_sources(self) -> List[str]:
        """Get list of available integrated sources."""
        if not self.integrated_path.exists():
            return []

        sources = []
        for source_dir in self.integrated_path.iterdir():
            if source_dir.is_dir() and not source_dir.name.startswith('.'):
                if any(source_dir.glob("*.jsonl")):
                    sources.append(source_dir.name)

        return sorted(sources)

    def load_samples(self, source: str, split: str) -> List[Dict[str, Any]]:
        """Load samples from a dataset split."""
        split_file = self.integrated_path / source / f"{split}.jsonl"

        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return []

        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        return samples

    def analyze_source(self, source: str, split: str) -> Dict[str, Any]:
        """Analyze coverage for a single source dataset."""
        samples = self.load_samples(source, split)
        total_samples = len(samples)

        if total_samples == 0:
            return {
                'source': source,
                'split': split,
                'total_samples': 0,
                'llm_models': [],
                'human_annotators': []
            }

        # Track per-model/annotator counts
        llm_counts = defaultdict(int)
        human_counts = defaultdict(int)

        samples_with_any_llm = 0
        samples_with_any_human = 0
        samples_with_both = 0

        for sample in samples:
            llm_scores = sample.get('llm_scores', []) or []
            human_scores = sample.get('human_scores', []) or []

            has_llm = len(llm_scores) > 0
            has_human = len(human_scores) > 0

            if has_llm:
                samples_with_any_llm += 1
                for score_entry in llm_scores:
                    model_name = score_entry.get('model_name')
                    if model_name:
                        llm_counts[model_name] += 1

            if has_human:
                samples_with_any_human += 1
                for score_entry in human_scores:
                    annotator_id = score_entry.get('annotator_id')
                    if annotator_id:
                        human_counts[annotator_id] += 1

            if has_llm and has_human:
                samples_with_both += 1

        # Build model coverage list
        llm_models = []
        for model_name, count in llm_counts.items():
            coverage_pct = (count / total_samples) * 100
            llm_models.append({
                'model_name': model_name,
                'samples_scored': count,
                'total_samples': total_samples,
                'coverage_percentage': round(coverage_pct, 2)
            })
        llm_models.sort(key=lambda x: -x['coverage_percentage'])

        # Build annotator coverage list
        human_annotators = []
        for annotator_id, count in human_counts.items():
            coverage_pct = (count / total_samples) * 100
            human_annotators.append({
                'annotator_id': annotator_id,
                'samples_scored': count,
                'total_samples': total_samples,
                'coverage_percentage': round(coverage_pct, 2)
            })
        human_annotators.sort(key=lambda x: -x['coverage_percentage'])

        return {
            'source': source,
            'split': split,
            'total_samples': total_samples,
            'samples_with_any_llm': samples_with_any_llm,
            'samples_with_any_human': samples_with_any_human,
            'samples_with_both': samples_with_both,
            'llm_coverage_percentage': round(samples_with_any_llm / total_samples * 100, 2) if total_samples > 0 else 0,
            'human_coverage_percentage': round(samples_with_any_human / total_samples * 100, 2) if total_samples > 0 else 0,
            'llm_models': llm_models,
            'human_annotators': human_annotators
        }

    def analyze(
        self,
        split: str,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main analysis entry point.

        Args:
            split: Dataset split to analyze ('train', 'valid', or 'test')
            sources: List of sources to analyze (if None, analyzes all)

        Returns:
            Comprehensive coverage analysis results
        """
        if sources is None:
            sources = self.get_available_sources()

        if not sources:
            logger.error("No sources found to analyze")
            return {'error': 'No sources found'}

        logger.info(f"Analyzing coverage for split: {split}")
        logger.info(f"Sources: {', '.join(sources)}")

        # Analyze each source
        source_results = []
        for source in sources:
            logger.info(f"Analyzing source: {source}")
            coverage = self.analyze_source(source, split)
            source_results.append(coverage)

        # Aggregate cross-source statistics
        all_llm_models = defaultdict(int)  # model_name -> samples_scored
        all_human_annotators = defaultdict(int)  # annotator_id -> samples_scored
        total_samples_all = 0

        for coverage in source_results:
            total_samples_all += coverage['total_samples']
            for model in coverage['llm_models']:
                all_llm_models[model['model_name']] += model['samples_scored']
            for annotator in coverage['human_annotators']:
                all_human_annotators[annotator['annotator_id']] += annotator['samples_scored']

        # Build aggregate model coverage (always use total_samples_all as denominator)
        aggregate_llm = []
        for model_name, samples_scored in all_llm_models.items():
            coverage_pct = (samples_scored / total_samples_all) * 100 if total_samples_all > 0 else 0
            aggregate_llm.append({
                'model_name': model_name,
                'samples_scored': samples_scored,
                'total_samples': total_samples_all,
                'coverage_percentage': round(coverage_pct, 2)
            })
        aggregate_llm.sort(key=lambda x: -x['coverage_percentage'])

        aggregate_human = []
        for annotator_id, samples_scored in all_human_annotators.items():
            coverage_pct = (samples_scored / total_samples_all) * 100 if total_samples_all > 0 else 0
            aggregate_human.append({
                'annotator_id': annotator_id,
                'samples_scored': samples_scored,
                'total_samples': total_samples_all,
                'coverage_percentage': round(coverage_pct, 2)
            })
        aggregate_human.sort(key=lambda x: -x['coverage_percentage'])

        return {
            'split': split,
            'sources_analyzed': len(source_results),
            'total_samples': total_samples_all,
            'summary': {
                'total_llm_models': len(aggregate_llm),
                'total_human_annotators': len(aggregate_human)
            },
            'llm_models': aggregate_llm,
            'human_annotators': aggregate_human,
            'per_source': source_results
        }

    def print_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted coverage report to console."""
        print("\n" + "="*70)
        print("COVERAGE ANALYSIS REPORT")
        print("="*70)
        print(f"Split: {results['split']}")
        print(f"Sources analyzed: {results['sources_analyzed']}")
        print(f"Total samples: {results['total_samples']}")

        # Summary
        summary = results['summary']
        print(f"\nTotal LLM models: {summary['total_llm_models']}")
        print(f"Total human annotators: {summary['total_human_annotators']}")

        # LLM Models
        if results['llm_models']:
            print("\n" + "-"*70)
            print("LLM MODEL COVERAGE (aggregate across all sources)")
            print("-"*70)
            print(f"{'Model':<45} {'Scored':<12} {'Coverage':<10}")
            print("-"*70)
            for model in results['llm_models']:
                print(f"{model['model_name']:<45} {model['samples_scored']}/{model['total_samples']:<10} {model['coverage_percentage']}%")

        # Human annotators
        if results['human_annotators']:
            print("\n" + "-"*70)
            print("HUMAN ANNOTATOR COVERAGE (aggregate across all sources)")
            print("-"*70)
            print(f"{'Annotator':<45} {'Scored':<12} {'Coverage':<10}")
            print("-"*70)
            for annotator in results['human_annotators']:
                print(f"{annotator['annotator_id']:<45} {annotator['samples_scored']}/{annotator['total_samples']:<10} {annotator['coverage_percentage']}%")

        # Per-source breakdown
        print("\n" + "-"*70)
        print("PER-SOURCE BREAKDOWN")
        print("-"*70)
        for source_data in results['per_source']:
            print(f"\n{source_data['source'].upper()} ({source_data['total_samples']} samples)")
            print(f"  Samples with LLM scores: {source_data['samples_with_any_llm']} ({source_data['llm_coverage_percentage']}%)")
            print(f"  Samples with human scores: {source_data['samples_with_any_human']} ({source_data['human_coverage_percentage']}%)")
            print(f"  Samples with both: {source_data['samples_with_both']}")

            if source_data['llm_models']:
                print(f"\n  LLM Models ({len(source_data['llm_models'])}):")
                for model in source_data['llm_models']:
                    print(f"    {model['model_name']}: {model['samples_scored']}/{model['total_samples']} ({model['coverage_percentage']}%)")

            if source_data['human_annotators']:
                print(f"\n  Human Annotators ({len(source_data['human_annotators'])}):")
                for annotator in source_data['human_annotators']:
                    print(f"    {annotator['annotator_id']}: {annotator['samples_scored']}/{annotator['total_samples']} ({annotator['coverage_percentage']}%)")

        print("\n" + "="*70)
