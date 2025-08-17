"""
CodeSearchNet dataset analyzer for comprehensive statistical analysis.

This module provides detailed analysis of the CodeSearchNet dataset to understand:
1. Lines of code distribution by language
2. Description length distribution by language  
3. Repository distribution by language
4. Function complexity patterns
5. Quality indicators for stratified sampling
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization libraries not available. Install matplotlib, seaborn, pandas, numpy for full functionality.")


logger = logging.getLogger(__name__)


class CodeSearchNetAnalyzer:
    """Comprehensive analyzer for CodeSearchNet dataset."""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize the analyzer.
        
        Args:
            dataset_path: Path to the converted CodeSearchNet dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.analysis_results = None
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    def load_dataset_split(self, file_path: Path) -> List[Dict]:
        """Load a dataset split (train/valid/test) from JSONL file."""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))
        
        return samples
    
    def analyze_code_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Basic complexity indicators
        complexity = {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'avg_line_length': np.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0,
            'max_line_length': max([len(line) for line in non_empty_lines]) if non_empty_lines else 0,
            'char_count': len(code),
            'has_loops': any(keyword in code.lower() for keyword in ['for ', 'while ', 'foreach']),
            'has_conditionals': any(keyword in code.lower() for keyword in ['if ', 'else', 'switch', 'case']),
            'has_try_catch': any(keyword in code.lower() for keyword in ['try', 'catch', 'except', 'finally']),
            'function_calls': code.count('('),  # Rough estimate
            'nested_blocks': code.count('{') + code.count('def ') + code.count('class '),  # Rough estimate
        }
        
        return complexity
    
    def analyze_description_quality(self, description: str) -> Dict[str, Any]:
        """Analyze description quality metrics."""
        if not description:
            return {'length': 0, 'word_count': 0, 'sentence_count': 0, 'quality_score': 0}
        
        words = description.split()
        sentences = description.split('.')
        
        quality = {
            'length': len(description),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'has_examples': 'example' in description.lower() or 'e.g.' in description.lower(),
            'has_parameters': any(keyword in description.lower() for keyword in ['param', 'arg', 'parameter']),
            'has_return': any(keyword in description.lower() for keyword in ['return', 'returns']),
            'quality_score': min(5, len(words) / 5)  # Simple quality heuristic
        }
        
        return quality
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Create comprehensive analysis of the CodeSearchNet dataset."""
        
        # Load all splits
        logger.info("Loading dataset splits...")
        splits_data = {}
        
        for split in ['train', 'valid', 'test']:
            split_file = self.dataset_path / f"{split}.jsonl"
            if split_file.exists():
                splits_data[split] = self.load_dataset_split(split_file)
                logger.info(f"Loaded {len(splits_data[split])} samples from {split} split")
        
        if not splits_data:
            raise FileNotFoundError(f"No dataset splits found in {self.dataset_path}")
        
        # Combine all data for comprehensive analysis
        all_samples = []
        for split, samples in splits_data.items():
            for sample in samples:
                sample['split'] = split
                all_samples.append(sample)
        
        logger.info(f"Total samples for analysis: {len(all_samples)}")
        
        # Initialize analysis results
        analysis = {
            'summary': {},
            'by_language': defaultdict(lambda: {
                'count': 0,
                'repositories': set(),
                'code_complexity': [],
                'description_quality': [],
                'lines_of_code': [],
                'description_lengths': [],
                'function_names': [],
                'repository_distribution': Counter()
            }),
            'overall_stats': {
                'total_samples': len(all_samples),
                'languages': set(),
                'repositories': set(),
                'complexity_distribution': [],
                'quality_distribution': []
            }
        }
        
        # Analyze each sample
        logger.info("Analyzing samples...")
        for i, sample in enumerate(all_samples):
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{len(all_samples)} samples")
            
            language = sample['metadata'].get('language', 'unknown')
            repo = sample['metadata'].get('repository_name', 'unknown')
            func_name = sample['metadata'].get('function_name', 'unknown')
            
            # Code analysis
            code = sample.get('submission', '')
            code_complexity = self.analyze_code_complexity(code)
            
            # Description analysis  
            description = sample.get('problem', '')
            desc_quality = self.analyze_description_quality(description)
            
            # Update language-specific stats
            lang_stats = analysis['by_language'][language]
            lang_stats['count'] += 1
            lang_stats['repositories'].add(repo)
            lang_stats['code_complexity'].append(code_complexity)
            lang_stats['description_quality'].append(desc_quality)
            lang_stats['lines_of_code'].append(code_complexity['non_empty_lines'])
            lang_stats['description_lengths'].append(desc_quality['length'])
            lang_stats['function_names'].append(func_name)
            lang_stats['repository_distribution'][repo] += 1
            
            # Update overall stats
            analysis['overall_stats']['languages'].add(language)
            analysis['overall_stats']['repositories'].add(repo)
            analysis['overall_stats']['complexity_distribution'].append(code_complexity['non_empty_lines'])
            analysis['overall_stats']['quality_distribution'].append(desc_quality['quality_score'])
        
        # Convert sets to lists for JSON serialization
        for lang in analysis['by_language']:
            analysis['by_language'][lang]['repositories'] = list(analysis['by_language'][lang]['repositories'])
        
        analysis['overall_stats']['languages'] = list(analysis['overall_stats']['languages'])
        analysis['overall_stats']['repositories'] = list(analysis['overall_stats']['repositories'])
        
        # Generate summary statistics
        analysis['summary'] = self.generate_summary_stats(analysis)
        
        self.analysis_results = analysis
        return analysis
    
    def generate_summary_stats(self, analysis: Dict) -> Dict:
        """Generate summary statistics from the analysis."""
        summary = {}
        
        for language, stats in analysis['by_language'].items():
            if stats['count'] == 0:
                continue
                
            lines_of_code = stats['lines_of_code']
            desc_lengths = stats['description_lengths']
            repo_counts = list(stats['repository_distribution'].values())
            
            summary[language] = {
                'total_samples': stats['count'],
                'unique_repositories': len(stats['repositories']),
                'lines_of_code_stats': {
                    'mean': np.mean(lines_of_code),
                    'median': np.median(lines_of_code),
                    'std': np.std(lines_of_code),
                    'min': np.min(lines_of_code),
                    'max': np.max(lines_of_code),
                    'q25': np.percentile(lines_of_code, 25),
                    'q75': np.percentile(lines_of_code, 75)
                },
                'description_length_stats': {
                    'mean': np.mean(desc_lengths),
                    'median': np.median(desc_lengths),
                    'std': np.std(desc_lengths),
                    'min': np.min(desc_lengths),
                    'max': np.max(desc_lengths),
                    'q25': np.percentile(desc_lengths, 25),
                    'q75': np.percentile(desc_lengths, 75)
                },
                'repository_stats': {
                    'total_repos': len(stats['repositories']),
                    'avg_samples_per_repo': np.mean(repo_counts),
                    'max_samples_per_repo': np.max(repo_counts),
                    'top_repositories': stats['repository_distribution'].most_common(10)
                }
            }
        
        return summary
    
    def create_visualizations(self, output_dir: Path):
        """Create comprehensive visualizations of the dataset analysis."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available. Skipping plot generation.")
            return
        
        if not self.analysis_results:
            raise ValueError("Run analysis first using run_comprehensive_analysis()")
        
        output_dir.mkdir(exist_ok=True)
        analysis = self.analysis_results
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        languages = list(analysis['by_language'].keys())
        
        # 1. Create main overview plot
        self._create_overview_plot(analysis, languages, output_dir)
        
        # 2. Create detailed plots for each language
        for lang in languages:
            if analysis['by_language'][lang]['count'] > 0:
                self._create_language_detailed_plot(analysis, lang, output_dir)
    
    def _create_overview_plot(self, analysis: Dict, languages: List[str], output_dir: Path):
        """Create the main 6-panel overview plot."""
        plt.figure(figsize=(15, 10))
        
        # 1. Lines of code distribution by language
        plt.subplot(2, 3, 1)
        lines_data = []
        for lang in languages:
            lines_data.extend([(lang, loc) for loc in analysis['by_language'][lang]['lines_of_code']])
        
        df_lines = pd.DataFrame(lines_data, columns=['Language', 'Lines_of_Code'])
        sns.boxplot(data=df_lines, x='Language', y='Lines_of_Code')
        plt.title('Lines of Code Distribution by Language')
        plt.xticks(rotation=45)
        plt.yscale('log')
        
        # 2. Description length distribution by language
        plt.subplot(2, 3, 2)
        desc_data = []
        for lang in languages:
            desc_data.extend([(lang, length) for length in analysis['by_language'][lang]['description_lengths']])
        
        df_desc = pd.DataFrame(desc_data, columns=['Language', 'Description_Length'])
        sns.boxplot(data=df_desc, x='Language', y='Description_Length')
        plt.title('Description Length Distribution by Language')
        plt.xticks(rotation=45)
        
        # 3. Repository count by language
        plt.subplot(2, 3, 3)
        repo_counts = [len(analysis['by_language'][lang]['repositories']) for lang in languages]
        plt.bar(languages, repo_counts)
        plt.title('Number of Repositories by Language')
        plt.xticks(rotation=45)
        
        # 4. Sample count by language
        plt.subplot(2, 3, 4)
        sample_counts = [analysis['by_language'][lang]['count'] for lang in languages]
        plt.bar(languages, sample_counts)
        plt.title('Sample Count by Language')
        plt.xticks(rotation=45)
        
        # 5. Average samples per repository
        plt.subplot(2, 3, 5)
        avg_samples_per_repo = []
        for lang in languages:
            repo_dist = analysis['by_language'][lang]['repository_distribution']
            avg_samples_per_repo.append(np.mean(list(repo_dist.values())) if repo_dist else 0)
        
        plt.bar(languages, avg_samples_per_repo)
        plt.title('Average Samples per Repository')
        plt.xticks(rotation=45)
        
        # 6. Lines of code vs description length scatter
        plt.subplot(2, 3, 6)
        for lang in languages:
            lines = analysis['by_language'][lang]['lines_of_code']
            desc_lens = analysis['by_language'][lang]['description_lengths']
            if lines and desc_lens:
                # Sample for visualization if too many points
                if len(lines) > 1000:
                    indices = np.random.choice(len(lines), 1000, replace=False)
                    lines = [lines[i] for i in indices]
                    desc_lens = [desc_lens[i] for i in indices]
                plt.scatter(lines, desc_lens, alpha=0.6, label=lang, s=20)
        
        plt.xlabel('Lines of Code')
        plt.ylabel('Description Length')
        plt.title('Code Length vs Description Length')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'codesearchnet_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_language_detailed_plot(self, analysis: Dict, lang: str, output_dir: Path):
        """Create detailed plot for a specific language."""
        plt.figure(figsize=(15, 5))
        
        # Lines of code histogram
        plt.subplot(1, 3, 1)
        lines = analysis['by_language'][lang]['lines_of_code']
        plt.hist(lines, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{lang}: Lines of Code Distribution')
        plt.xlabel('Lines of Code')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        # Description length histogram  
        plt.subplot(1, 3, 2)
        desc_lens = analysis['by_language'][lang]['description_lengths']
        plt.hist(desc_lens, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{lang}: Description Length Distribution')
        plt.xlabel('Description Length (chars)')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        # Top repositories
        plt.subplot(1, 3, 3)
        top_repos = analysis['by_language'][lang]['repository_distribution'].most_common(10)
        if top_repos:
            repos, counts = zip(*top_repos)
            repos = [repo.split('/')[-1][:15] + '...' if len(repo) > 15 else repo for repo in repos]
            plt.barh(range(len(repos)), counts)
            plt.yticks(range(len(repos)), repos)
            plt.title(f'{lang}: Top 10 Repositories')
            plt.xlabel('Sample Count')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{lang}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: Path):
        """Save analysis results to files."""
        if not self.analysis_results:
            raise ValueError("Run analysis first using run_comprehensive_analysis()")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis = self.analysis_results
        
        # Save full analysis as JSON
        with open(output_dir / 'full_analysis.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_analysis = json.loads(json.dumps(analysis, default=str))
            json.dump(json_analysis, f, indent=2)
        
        # Save summary as Markdown
        self._save_markdown_report(analysis, output_dir)
        
        logger.info(f"Analysis results saved to {output_dir}")
    
    def _save_markdown_report(self, analysis: Dict, output_dir: Path):
        """Save comprehensive Markdown report."""
        with open(output_dir / 'analysis_summary.md', 'w') as f:
            f.write("# CodeSearchNet Dataset Analysis Summary\n\n")
            
            # Overview section
            f.write("## Dataset Overview\n\n")
            f.write("This analysis covers the stratified split CodeSearchNet dataset with balanced language representation.\n\n")
            
            # Overview table
            f.write("| Language | Samples | Repositories | Avg Lines/Code | Avg Desc Length | Med Lines/Code | Med Desc Length |\n")
            f.write("|----------|---------|--------------|----------------|-----------------|----------------|------------------|\n")
            
            for language, stats in analysis['summary'].items():
                f.write(f"| {language.title()} | {stats['total_samples']:,} | "
                       f"{stats['unique_repositories']:,} | "
                       f"{stats['lines_of_code_stats']['mean']:.1f} | "
                       f"{stats['description_length_stats']['mean']:.1f} | "
                       f"{stats['lines_of_code_stats']['median']:.1f} | "
                       f"{stats['description_length_stats']['median']:.1f} |\n")
            
            f.write("\n")
            
            # Complexity distribution summary
            f.write("## Code Complexity Overview\n\n")
            f.write("| Language | Simple (≤10 lines) | Medium (11-25 lines) | Complex (26-50 lines) | Very Complex (>50 lines) |\n")
            f.write("|----------|-------------------|----------------------|----------------------|-------------------------|\n")
            
            for language, stats in analysis['by_language'].items():
                lines = stats['lines_of_code']
                simple = sum(1 for x in lines if x <= 10)
                medium = sum(1 for x in lines if 10 < x <= 25)
                complex_code = sum(1 for x in lines if 25 < x <= 50)
                very_complex = sum(1 for x in lines if x > 50)
                total = len(lines)
                
                f.write(f"| {language.title()} | {simple:,} ({simple/total*100:.1f}%) | "
                       f"{medium:,} ({medium/total*100:.1f}%) | "
                       f"{complex_code:,} ({complex_code/total*100:.1f}%) | "
                       f"{very_complex:,} ({very_complex/total*100:.1f}%) |\n")
            
            f.write("\n")
            
            # Repository concentration
            f.write("## Repository Concentration\n\n")
            f.write("| Language | Total Repos | Avg Samples/Repo | Max Samples/Repo | Top Repo % |\n")
            f.write("|----------|-------------|------------------|------------------|-----------|\n")
            
            for language, stats in analysis['summary'].items():
                repo_stats = stats['repository_stats']
                top_repo_count = repo_stats['top_repositories'][0][1] if repo_stats['top_repositories'] else 0
                top_repo_percent = (top_repo_count / stats['total_samples']) * 100
                
                f.write(f"| {language.title()} | {repo_stats['total_repos']:,} | "
                       f"{repo_stats['avg_samples_per_repo']:.1f} | "
                       f"{repo_stats['max_samples_per_repo']:,} | "
                       f"{top_repo_percent:.1f}% |\n")
            
            f.write("\n")
            
            # Detailed sections for each language
            for language, stats in analysis['summary'].items():
                f.write(f"## {language.title()} Detailed Analysis\n\n")
                
                f.write(f"**Total Samples:** {stats['total_samples']:,}  \n")
                f.write(f"**Unique Repositories:** {stats['unique_repositories']:,}  \n")
                f.write(f"**Samples per Repository:** {stats['repository_stats']['avg_samples_per_repo']:.1f} average\n\n")
                
                # Code complexity table
                f.write("### Lines of Code Distribution\n\n")
                f.write("| Metric | Value |\n|--------|-------|\n")
                loc_stats = stats['lines_of_code_stats']
                f.write(f"| Mean | {loc_stats['mean']:.1f} |\n")
                f.write(f"| Median | {loc_stats['median']:.1f} |\n")
                f.write(f"| Standard Deviation | {loc_stats['std']:.1f} |\n")
                f.write(f"| Q25 - Q75 | {loc_stats['q25']:.1f} - {loc_stats['q75']:.1f} |\n")
                f.write(f"| Range | {loc_stats['min']} - {loc_stats['max']} |\n\n")
                
                # Description quality table  
                f.write("### Description Length Distribution\n\n")
                f.write("| Metric | Value |\n|--------|-------|\n")
                desc_stats = stats['description_length_stats']
                f.write(f"| Mean | {desc_stats['mean']:.1f} chars |\n")
                f.write(f"| Median | {desc_stats['median']:.1f} chars |\n")
                f.write(f"| Standard Deviation | {desc_stats['std']:.1f} chars |\n")
                f.write(f"| Q25 - Q75 | {desc_stats['q25']:.1f} - {desc_stats['q75']:.1f} chars |\n")
                f.write(f"| Range | {desc_stats['min']} - {desc_stats['max']} chars |\n\n")
                
                # Top repositories
                f.write("### Top 10 Repositories\n\n")
                f.write("| Repository | Sample Count | Percentage |\n|------------|-------------|------------|\n")
                for repo, count in stats['repository_stats']['top_repositories']:
                    percentage = (count / stats['total_samples']) * 100
                    f.write(f"| `{repo}` | {count:,} | {percentage:.1f}% |\n")
                
                f.write("\n")
            
            # Visualizations section
            f.write("## Dataset Visualizations\n\n")
            f.write("### Overview Analysis\n\n")
            f.write("![CodeSearchNet Dataset Overview](codesearchnet_analysis.png)\n\n")
            f.write("*Six-panel overview showing distribution patterns across languages*\n\n")
            
            f.write("### Per-Language Detailed Analysis\n\n")
            for language in analysis['by_language'].keys():
                f.write(f"#### {language.title()} Language Analysis\n\n")
                f.write(f"![{language.title()} Detailed Analysis]({language}_detailed_analysis.png)\n\n")
            
            # Sampling recommendations
            f.write("## Sampling Strategy Recommendations\n\n")
            f.write("Based on this analysis, for human annotation of 100 samples per language:\n\n")
            
            f.write("### 1. Code Complexity Stratification\n")
            f.write("- **Simple functions (≤10 lines):** 25 samples\n")
            f.write("- **Medium functions (11-25 lines):** 40 samples\n") 
            f.write("- **Complex functions (26-50 lines):** 25 samples\n")
            f.write("- **Very complex functions (>50 lines):** 10 samples\n\n")
            
            f.write("### 2. Repository Diversity\n")
            f.write("- Limit to max 5 samples per repository\n")
            f.write("- Prioritize repositories with moderate activity (not too dominant, not too rare)\n")
            f.write("- Ensure coverage of different project types\n\n")
            
            f.write("### 3. Description Quality\n")
            f.write("- Include mix of well-documented and poorly-documented functions\n")
            f.write("- Prioritize functions with meaningful descriptions (>50 characters)\n")
            f.write("- Avoid functions with trivial descriptions\n\n")
            
            f.write("### 4. Quality Control\n")
            f.write("- Pre-filter auto-generated code\n")
            f.write("- Remove obvious getters/setters\n")
            f.write("- Exclude test functions\n")
            f.write("- Ensure functions are self-contained\n\n")
    
    def analyze(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            output_dir: Directory to save results. If None, uses default path.
            
        Returns:
            Analysis results dictionary
        """
        if output_dir is None:
            output_dir = Path("generated") / "analysis" / "codesearchnet"
        
        logger.info(f"Starting comprehensive analysis of CodeSearchNet dataset at {self.dataset_path}")
        
        # Run analysis
        analysis = self.run_comprehensive_analysis()
        
        # Save results
        logger.info("Saving analysis results...")
        self.save_results(output_dir)
        
        # Create visualizations
        if VISUALIZATION_AVAILABLE:
            logger.info("Creating visualizations...")
            self.create_visualizations(output_dir)
        else:
            logger.warning("Skipping visualizations (libraries not available)")
        
        logger.info(f"Analysis complete! Results saved to {output_dir}")
        logger.info("Check the following files:")
        logger.info(f"  - {output_dir}/analysis_summary.md (comprehensive Markdown report)")
        logger.info(f"  - {output_dir}/full_analysis.json (complete data)")
        if VISUALIZATION_AVAILABLE:
            logger.info(f"  - {output_dir}/codesearchnet_analysis.png (overview plots)")
            logger.info(f"  - {output_dir}/*_detailed_analysis.png (per-language plots)")
        
        return analysis