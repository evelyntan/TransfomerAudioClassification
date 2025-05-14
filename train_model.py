import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize wandb
        wandb.init(
            project=config.project_name,
            config=config.__dict__,
            name=config.run_name
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.patience_counter = 0
        
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            wandb.save(str(best_model_path))
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(self):
        vocab = {
            0:"air_conditioner",
            1:"car_horn",
            2:"children_playing",
            3:"dog_bark",
            4:"drilling",
            5:"engine_idling",
            6:"gun_shot",
            7:"jackhammer",
            8:"siren",
            9:"street_music"
        }
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Training phase with its own progress bar
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Train]')
            for batch_idx, (audio, labels) in enumerate(train_pbar):
                # Move data to device
                audio = audio.squeeze(1).to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(audio)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += loss.item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': f'{100. * correct / total:.2f}%'
                })
            
            # Calculate training metrics
            train_loss = total_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase with its own progress bar
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs} [Val]')
                for audio, labels in val_pbar:
                    # Move data to device
                    audio = audio.squeeze(1).to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(audio)
                    loss = self.criterion(outputs, labels)
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_loss += loss.item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': val_loss / (batch_idx + 1),
                        'acc': f'{100. * val_correct / val_total:.2f}%'
                    })
            
                # Calculate validation metrics
                val_loss = val_loss / len(self.val_loader)
                val_acc = 100. * val_correct / val_total

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            

                
            # Sample prediction
            self.model.eval()
            with torch.no_grad():
                # Get a random sample from validation set
                sample_audio, sample_label = next(iter(self.val_loader))
                sample_audio = sample_audio.squeeze(1).to(self.device)
                
                # Make prediction
                output = self.model(sample_audio)
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                # Print sample predictions
                print(f"\nEpoch {epoch+1} Sample Predictions:")
                for i in range(min(2, len(sample_label))):  # Show first 2 samples
                    print(f"\nSample {i+1}:")
                    print(f"True Label: {sample_label[i]}")
                    print(f"Predicted Label: {vocab[predicted[i].item()]}")
                    print(f"Confidence: {probabilities[i][predicted[i]].item():.2%}")
                    
                    # Print top 3 predictions
                    top3_prob, top3_indices = torch.topk(probabilities[i], 3)
                    print("Top 3 predictions:")
                    for prob, idx in zip(top3_prob, top3_indices):
                        print(f"{vocab[idx.item()]}: {prob.item():.2%}")
            
            # Update learning rate
            self.scheduler.step(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        wandb.finish()

# Configuration class
class Config:
    def __init__(self):
        self.project_name = "audio-classification"
        self.run_name = "experiment-1"
        self.checkpoint_dir = "checkpoints"
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 10
        self.patience = 10  # for early stopping
