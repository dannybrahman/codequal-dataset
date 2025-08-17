"""
CodeQual Dataset Viewer - Flask Web Application

A web interface for exploring and visualizing CodeQual datasets with
filtering capabilities and syntax highlighting.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound
import markdown

from data_loader import DatasetLoader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Global dataset loader (initialized on startup)
dataset_loader = None


def get_code_highlighted(code: str, language: str) -> str:
    """Apply syntax highlighting to code."""
    try:
        # Map common language names to Pygments lexer names
        language_map = {
            'javascript': 'javascript',
            'python': 'python',
            'java': 'java',
            'go': 'go',
            'php': 'php',
            'ruby': 'ruby',
            'cpp': 'cpp',
            'c++': 'cpp'
        }
        
        lexer_name = language_map.get(language.lower(), language.lower())
        lexer = get_lexer_by_name(lexer_name)
    except ClassNotFound:
        # Fallback to guessing lexer from code
        try:
            lexer = guess_lexer(code)
        except:
            # Final fallback to plain text
            lexer = get_lexer_by_name('text')
    
    # Try to use a modern style, fallback to default if not available
    try:
        formatter = HtmlFormatter(style='colorful', cssclass='highlight')
    except:
        try:
            formatter = HtmlFormatter(style='tango', cssclass='highlight')
        except:
            formatter = HtmlFormatter(style='default', cssclass='highlight')
    
    highlighted = highlight(code, lexer, formatter)
    return highlighted


def get_problem_formatted(problem_text: str) -> str:
    """Format problem description (supports basic markdown)."""
    # Convert basic markdown to HTML
    html = markdown.markdown(problem_text, extensions=['nl2br', 'fenced_code'])
    return html


@app.route('/')
def index():
    """Main dataset viewer page."""
    if not dataset_loader:
        return render_template('error.html', 
                             message="Dataset not loaded. Check DATASET_PATH configuration.")
    
    # Get filter options and stats
    filter_options = dataset_loader.get_filter_options()
    stats = dataset_loader.get_dataset_stats()
    
    # Get initial samples (first 20)
    samples, total_count = dataset_loader.get_samples(limit=20)
    
    return render_template('index.html',
                         samples=samples,
                         total_count=total_count,
                         filter_options=filter_options,
                         stats=stats,
                         get_code_highlighted=get_code_highlighted,
                         get_problem_formatted=get_problem_formatted)


@app.route('/api/samples')
def api_samples():
    """API endpoint for filtered samples with pagination."""
    if not dataset_loader:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    # Parse query parameters
    split = request.args.get('split')
    language = request.args.get('language')
    
    # Lines of code filtering
    min_lines = request.args.get('min_lines', type=int)
    max_lines = request.args.get('max_lines', type=int)
    
    # Description length filtering
    min_description = request.args.get('min_description', type=int)
    max_description = request.args.get('max_description', type=int)
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    offset = (page - 1) * per_page
    
    # Random sampling
    random_seed = request.args.get('random_seed', type=int)
    
    try:
        # Get filtered samples
        samples, total_count = dataset_loader.get_samples(
            split=split,
            language=language,
            min_lines=min_lines,
            max_lines=max_lines,
            min_description=min_description,
            max_description=max_description,
            limit=per_page,
            offset=offset,
            random_seed=random_seed
        )
        
        # Convert samples to dictionary format
        samples_data = []
        for sample in samples:
            samples_data.append({
                'problem_id': sample.problem_id,
                'submission_id': sample.submission_id,
                'problem': sample.problem,
                'problem_formatted': get_problem_formatted(sample.problem),
                'submission': sample.submission,
                'submission_highlighted': get_code_highlighted(sample.submission, sample.language),
                'source': sample.source,
                'split': sample.split,
                'language': sample.language,
                'lines_of_code': sample.lines_of_code,
                'description_length': sample.description_length,
                'repository': sample.repository,
                'function_name': sample.function_name,
                'quality_scores': sample.quality_scores
            })
        
        return jsonify({
            'samples': samples_data,
            'total_count': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page
        })
        
    except Exception as e:
        logger.error(f"Error fetching samples: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample/<submission_id>')
def api_sample_detail(submission_id: str):
    """API endpoint for individual sample details."""
    if not dataset_loader:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    sample = dataset_loader.get_sample_by_id(submission_id)
    if not sample:
        return jsonify({'error': 'Sample not found'}), 404
    
    return jsonify({
        'problem_id': sample.problem_id,
        'submission_id': sample.submission_id,
        'problem': sample.problem,
        'problem_formatted': get_problem_formatted(sample.problem),
        'submission': sample.submission,
        'submission_highlighted': get_code_highlighted(sample.submission, sample.language),
        'source': sample.source,
        'split': sample.split,
        'language': sample.language,
        'lines_of_code': sample.lines_of_code,
        'description_length': sample.description_length,
        'repository': sample.repository,
        'function_name': sample.function_name,
        'quality_scores': sample.quality_scores
    })


@app.route('/api/stats')
def api_stats():
    """API endpoint for dataset statistics."""
    if not dataset_loader:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    stats = dataset_loader.get_dataset_stats()
    filter_options = dataset_loader.get_filter_options()
    
    return jsonify({
        'stats': stats,
        'filter_options': filter_options
    })


@app.route('/sample/<submission_id>')
def sample_detail(submission_id: str):
    """Detailed view for a single sample."""
    if not dataset_loader:
        return render_template('error.html', 
                             message="Dataset not loaded. Check DATASET_PATH configuration.")
    
    sample = dataset_loader.get_sample_by_id(submission_id)
    if not sample:
        return render_template('error.html', 
                             message=f"Sample not found: {submission_id}")
    
    return render_template('sample_detail.html',
                         sample=sample,
                         get_code_highlighted=get_code_highlighted,
                         get_problem_formatted=get_problem_formatted)


def create_app(dataset_path: str = None):
    """Create and configure the Flask application."""
    global dataset_loader
    
    if dataset_path is None:
        dataset_path = os.getenv('DATASET_PATH')
        
    if not dataset_path:
        logger.error("DATASET_PATH environment variable not set")
        return app
    
    try:
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset_loader = DatasetLoader(dataset_path)
        logger.info("Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        dataset_loader = None
    
    return app


if __name__ == '__main__':
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Create app with dataset
    app = create_app()
    
    # Run the app
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting CodeQual Dataset Viewer on {host}:{port}")
    app.run(host=host, port=port, debug=debug)