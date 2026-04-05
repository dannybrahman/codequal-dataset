"""
Model Training CLI

Train and evaluate smaller models on LLM-generated synthetic scores.
"""

import logging
import argparse
import sys
import time
from pathlib import Path

import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import QualityDatasetLoader, QualityDataset, CodeFeatureExtractor
from src.models import get_model
from src.training import QualityModelTrainer, QualityLoss
from src.evaluation import ModelEvaluator


def create_session_dir(base_dir: str = "generated") -> Path:
    """
    Create a new session directory with unique ID.

    Structure:
        generated/run_session_<timestamp>/
            models/      - saved model checkpoints
            results/     - evaluation results
            logs/        - training logs
            config.json  - session configuration

    Returns:
        Path to session directory
    """
    session_id = f"run_session_{int(time.time())}"
    session_dir = Path(base_dir) / session_id

    # Create subdirectories
    (session_dir / "models").mkdir(parents=True, exist_ok=True)
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    (session_dir / "logs").mkdir(parents=True, exist_ok=True)

    return session_dir


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def train_command(args):
    """Train a model."""
    logger = logging.getLogger(__name__)

    # Create session directory
    session_dir = create_session_dir()
    models_dir = session_dir / "models"

    logger.info("=" * 80)
    logger.info("MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Session: {session_dir.name}")
    logger.info(f"Model type: {args.model}")
    logger.info(f"Sources: {', '.join(args.sources)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")

    # Load datasets
    logger.info("\nLoading datasets...")
    loader = QualityDatasetLoader()

    train_samples = loader.load_multi_source(args.sources, 'train')
    valid_samples = loader.load_multi_source(args.sources, 'valid')

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Valid samples: {len(valid_samples)}")

    # Initialize feature extractor
    if args.model == 'codebert':
        feature_extractor = CodeFeatureExtractor(method=args.feature_method, device=args.device)
    else:  # mlp
        feature_extractor = CodeFeatureExtractor(method='simple')

    # Create datasets
    train_dataset = QualityDataset(train_samples, feature_extractor)
    valid_dataset = QualityDataset(valid_samples, feature_extractor)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for MPS compatibility
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    logger.info("\nInitializing model...")
    if args.model == 'codebert':
        model = get_model('codebert',
                         hidden_dim=256,
                         dropout=0.1,
                         freeze_encoder=args.freeze_encoder)
    else:  # mlp
        model = get_model('mlp',
                         input_dim=21,  # Simple features
                         hidden_dims=[128, 64, 32],
                         dropout=0.2)

    # Initialize trainer
    loss_fn = QualityLoss(loss_type='mse')
    trainer = QualityModelTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        learning_rate=args.lr,
        device=args.device,
        output_dir=str(models_dir)
    )

    # Train
    logger.info("\nStarting training...")
    results = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_best=True
    )

    # Save model info
    feature_method = args.feature_method if args.model == 'codebert' else 'simple'
    training_config = {
        'feature_method': feature_method,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'patience': args.patience,
        'sources': args.sources,
        'train_samples': len(train_samples),
        'valid_samples': len(valid_samples),
    }
    trainer.save_model_info(training_config)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Session: {session_dir}")
    logger.info(f"Best validation loss: {results['best_valid_loss']:.4f}")
    logger.info(f"Total epochs: {results['total_epochs']}")
    logger.info(f"Total time: {results['total_time']:.2f}s")
    logger.info(f"Model saved to: {models_dir}/best_model.pt")

    return 0


def evaluate_command(args):
    """Evaluate a trained model."""
    logger = logging.getLogger(__name__)

    # Get session directory
    session_dir = Path("generated") / args.session
    if not session_dir.exists():
        logger.error(f"Session directory not found: {session_dir}")
        return 1

    models_dir = session_dir / "models"
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Default checkpoint path
    checkpoint_path = models_dir / "best_model.pt"

    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Session: {args.session}")
    logger.info(f"Model path: {checkpoint_path}")
    logger.info(f"Sources: {', '.join(args.sources)}")
    logger.info(f"Human score aggregation: {args.aggregation}")

    # Load test data
    logger.info("\nLoading test data...")
    loader = QualityDatasetLoader(human_score_aggregation=args.aggregation)
    test_samples = loader.load_multi_source(args.sources, 'test')

    logger.info(f"Test samples: {len(test_samples)}")

    # Load model
    logger.info("\nLoading model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']

    # Initialize model architecture
    if model_config['model_type'] == 'codebert':
        from src.models import CodeBERTRegressor
        model = CodeBERTRegressor(**{k: v for k, v in model_config.items()
                                    if k in ['model_name', 'hidden_dim', 'dropout', 'freeze_encoder']})
        feature_extractor = CodeFeatureExtractor(method='codebert', device=args.device)
    else:  # mlp
        from src.models import SimpleMLP
        model = SimpleMLP(**{k: v for k, v in model_config.items()
                           if k in ['input_dim', 'hidden_dims', 'dropout']})
        feature_extractor = CodeFeatureExtractor(method='simple')

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")

    # Create test dataset and loader
    test_dataset = QualityDataset(test_samples, feature_extractor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=args.device,
        output_dir=str(results_dir)
    )

    evaluator.evaluate_and_save()

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_dir}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate code quality models'
    )

    # Global arguments
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', required=True,
                             choices=['codebert', 'mlp'],
                             help='Model type')
    train_parser.add_argument('--sources', nargs='+',
                             default=['codenet', 'codeeval', 'codesearchnet', 'humaneval-x', 'mbpp'],
                             help='Dataset sources')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=20,
                             help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_parser.add_argument('--patience', type=int, default=4,
                             help='Early stopping patience')
    train_parser.add_argument('--freeze-encoder', action='store_true',
                             help='Freeze encoder weights (CodeBERT only)')
    train_parser.add_argument('--feature-method', default='codebert',
                             choices=['codebert', 'hybrid'],
                             help='Feature extraction method for CodeBERT (codebert=768, hybrid=768+21)')
    train_parser.add_argument('--device', default=None,
                             help='Device (cuda/mps/cpu or auto)')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--session', required=True,
                            help='Session directory name (e.g., run_session_1234567890)')
    eval_parser.add_argument('--sources', nargs='+',
                            default=['codenet', 'codeeval', 'codesearchnet', 'humaneval-x', 'mbpp'],
                            help='Dataset sources')
    eval_parser.add_argument('--aggregation', choices=['mean', 'median'], default='median',
                            help='Aggregation method for human scores (default: median)')
    eval_parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size')
    eval_parser.add_argument('--device', default=None,
                            help='Device (cuda/mps/cpu or auto)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    # Run command
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'evaluate':
        return evaluate_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
