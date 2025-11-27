"""
Model Trainer

Main training loop for quality prediction models.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ..models.base_model import BaseQualityModel
from .loss_functions import QualityLoss

logger = logging.getLogger(__name__)


class QualityModelTrainer:
    """
    Trainer for quality prediction models.

    Handles training loop, validation, checkpointing, and logging.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    """

    def __init__(
        self,
        model: BaseQualityModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        output_dir: str = "generated/models"
    ):
        """
        Initialize trainer.

        Args:
            model: Quality prediction model
            train_loader: Training data loader
            valid_loader: Validation data loader
            loss_fn: Loss function (default: MSE)
            learning_rate: Learning rate
            device: Device ('cuda', 'mps', 'cpu', or None for auto)
            output_dir: Directory for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Set device with MPS support for Apple Silicon
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Loss function
        if loss_fn is None:
            self.loss_fn = QualityLoss(loss_type='mse')
        else:
            self.loss_fn = loss_fn

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # Training state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_valid_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []

    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_features, batch_targets, _ in pbar:  # Ignore source during training
            # Move to device
            batch_targets = batch_targets.to(self.device)

            # Move features to device (handle dict)
            if isinstance(batch_features, dict):
                batch_features = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_features.items()
                }
            else:
                batch_features = batch_features.to(self.device)

            # Forward pass
            predictions = self.model(batch_features)
            loss = self.loss_fn(predictions, batch_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        return {'loss': avg_loss}

    def validate(self) -> Dict:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_features, batch_targets, _ in tqdm(self.valid_loader, desc="Validation"):  # Ignore source
                # Move to device
                batch_targets = batch_targets.to(self.device)

                if isinstance(batch_features, dict):
                    batch_features = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_features.items()
                    }
                else:
                    batch_features = batch_features.to(self.device)

                # Forward pass
                predictions = self.model(batch_features)
                loss = self.loss_fn(predictions, batch_targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        return {'loss': avg_loss}

    def train(
        self,
        num_epochs: int = 10,
        early_stopping_patience: int = 5,
        save_best: bool = True
    ) -> Dict:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs without improvement before stopping
            save_best: Whether to save best model checkpoint

        Returns:
            Training history dict
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train loss: {train_metrics['loss']:.4f}")

            # Validate
            valid_metrics = self.validate()
            logger.info(f"Valid loss: {valid_metrics['loss']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(valid_metrics['loss'])

            # Track history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'valid_loss': valid_metrics['loss'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

            # Save best model
            if valid_metrics['loss'] < self.best_valid_loss:
                self.best_valid_loss = valid_metrics['loss']
                self.epochs_without_improvement = 0

                if save_best:
                    self.save_checkpoint('best_model.pt', epoch=epoch)
                    logger.info(f"Saved best model (valid loss: {self.best_valid_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epochs")

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_valid_loss:.4f}")

        # Save final model and history
        self.save_checkpoint('final_model.pt', epoch=num_epochs)
        self.save_training_history()

        return {
            'best_valid_loss': self.best_valid_loss,
            'total_epochs': epoch,
            'total_time': total_time,
            'history': self.training_history,
            'trainer': self  # Return trainer for saving model info
        }

    def save_checkpoint(self, filename: str, **kwargs):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename

        self.model.save_checkpoint(
            str(checkpoint_path),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            best_valid_loss=self.best_valid_loss,
            training_history=self.training_history,
            **kwargs
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.output_dir / filename

        checkpoint = self.model.load_checkpoint(str(checkpoint_path))

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'best_valid_loss' in checkpoint:
            self.best_valid_loss = checkpoint['best_valid_loss']

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        logger.info(f"Loaded checkpoint: {checkpoint_path}")

        return checkpoint

    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"Saved training history: {history_path}")

    def save_model_info(self, training_config: dict = None):
        """Save model information to JSON."""
        model_info_path = self.output_dir / 'model.json'

        model_config = self.model.get_config()

        model_info = {
            'model_name': self.model.__class__.__name__,
            'model_type': model_config.get('model_type', 'unknown'),
            'num_parameters': self.model.get_num_parameters(),
            'model_config': model_config,
            'training_config': training_config or {},
            'best_valid_loss': self.best_valid_loss,
            'device': self.device,
        }

        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Saved model info: {model_info_path}")
