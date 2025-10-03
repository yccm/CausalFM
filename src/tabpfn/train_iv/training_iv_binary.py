# Training for IV setting with binary Z

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd  

import random
from typing import Dict, List, Tuple, Optional
import logging
from matplotlib import pyplot as plt
import time  
from datetime import datetime, timedelta  
import sys
import os

from src.tabpfn.model.causalFM4IV import PerFeatureTransformerCATE
from DATA_IV.causaldatasetIV import create_causal_data_loaders


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  


def format_time(seconds):
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:.0f}h {minutes:.0f}m {secs:.2f}s"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train IV settingCATE estimation model with CausalFM')
    parser.add_argument('--data_path', type=str, default="DATA_IV/binary_Z/synthetic*.csv",
                        help='Path to dataset CSV files')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Proportion of data to use for validation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps for learning rate scheduler')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (epochs)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging training progress')
    parser.add_argument('--save_dir', type=str, default='checkpoints_IV_binary',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save tensorboard logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use if available')
    parser.add_argument('--use_gmm_head', type=bool, default=True,
                        help='Use GMM head')
    parser.add_argument('--gmm_n_components', type=int, default=5,
                        help='Number of components in GMM')
    parser.add_argument('--gmm_min_sigma', type=float, default=1e-3,
                        help='Minimum sigma for GMM')
    parser.add_argument('--gmm_pi_temp', type=float, default=1.0,
                        help='Temperature for GMM')
    
    return parser.parse_args()


class CATETrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
        
        self.model = PerFeatureTransformerCATE(use_gmm_head=self.args.use_gmm_head,
                                               gmm_n_components=self.args.gmm_n_components,
                                               gmm_min_sigma=self.args.gmm_min_sigma,
                                               gmm_pi_temp=self.args.gmm_pi_temp)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.train_loader, self.val_loader, _ = create_causal_data_loaders(
            args.data_path, 
            batch_size=args.batch_size,
            val_split=args.val_split,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_cate_values = []
        self.best_val_loss = float('inf')
        
        self.epoch_times = []
        self.total_training_time = 0
        self.data_loading_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.validation_times = []

        self.eps = 1e-12 


    def gmm_nll_loss(self, pi, mu, sigma, target):
        """
        Compute negative log-likelihood for GMM
        Args:
        pi: mixture weights 
        mu: means 
        sigma: standard deviations 
        target: target values 
    """
        if target.dim() == 2:
            target = target.squeeze(-1)  

        pi = torch.clamp(pi, min=self.eps)                    
        pi = pi / pi.sum(dim=-1, keepdim=True)  
        sigma = torch.clamp(sigma, min=self.eps)             

        log_norm_const = 0.5 * np.log(2.0 * np.pi)
        z = (target - mu) / sigma
        log_prob = -0.5 * z*z - torch.log(sigma) - log_norm_const  
        log_mix = torch.logsumexp(torch.log(pi) + log_prob, dim=-1)  

        return -log_mix.mean()

            

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        epoch_start_time = time.time()
        
        batch_times = []
        data_load_times = []
        forward_times = []
        backward_times = []
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]") as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch_start_time = time.time()
                
                data_load_time = time.time() - batch_start_time

                x = batch['X'].to(self.device)  # Covariates 
                a = batch['a'].to(self.device)  # Treatment
                y = batch['y'].to(self.device)  # Factual outcome
                z = batch['z'].to(self.device)  # Instrumental variable
        
                y0 = batch['y0'].to(self.device)
                y1 = batch['y1'].to(self.device)
                ite = batch['ite'].to(self.device)  # True ite value

                single_eval_pos = int(len(x) * 0.8) 

                forward_start_time = time.time()
                out = self.model(x, a, y, z, single_eval_pos)  
                forward_time = time.time() - forward_start_time

                # Get GMM parameters
                gmm_pi = out['gmm_pi']
                gmm_mu = out['gmm_mu']
                gmm_sigma = out['gmm_sigma']

                ite_bs1 = ite.permute(1, 0, 2)  

                loss = self.gmm_nll_loss(gmm_pi, gmm_mu, gmm_sigma, ite_bs1)
                cate = (gmm_pi * gmm_mu).sum(dim=-1).unsqueeze(-1)  
                cate_diff = cate - ite_bs1
                
                # Backpropagation
                backward_start_time = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                backward_time = time.time() - backward_start_time
                
                total_loss += loss.item()
                
                if epoch == 0:
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    data_load_times.append(data_load_time)
                    forward_times.append(forward_time)
                    backward_times.append(backward_time)
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'batch_time': f"{time.time() - batch_start_time:.2f}s"
                })
                
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Log detailed timing for first epoch
        if epoch == 0:
            avg_batch_time = np.mean(batch_times)
            avg_data_load_time = np.mean(data_load_times)
            avg_forward_time = np.mean(forward_times)
            avg_backward_time = np.mean(backward_times)
            
            logger.info(f"First epoch detailed timing:")
            logger.info(f"  Average batch time: {avg_batch_time:.3f}s")
            logger.info(f"  Average data loading time: {avg_data_load_time:.3f}s ({avg_data_load_time/avg_batch_time*100:.1f}%)")
            logger.info(f"  Average forward pass time: {avg_forward_time:.3f}s ({avg_forward_time/avg_batch_time*100:.1f}%)")
            logger.info(f"  Average backward pass time: {avg_backward_time:.3f}s ({avg_backward_time/avg_batch_time*100:.1f}%)")
            
            # Store for overall tracking
            self.data_loading_time += avg_data_load_time * len(self.train_loader)
            self.forward_time += avg_forward_time * len(self.train_loader)
            self.backward_time += avg_backward_time * len(self.train_loader)
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        logger.info(f"Epoch {epoch+1} training completed in {format_time(epoch_time)}")
        return avg_loss

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_samples = 0
        sum_sq_err = 0.0 
        
        val_start_time = time.time()
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    x = batch['X'].to(self.device)
                    a = batch['a'].to(self.device)
                    y = batch['y'].to(self.device)
                    z = batch['z'].to(self.device)
                    ite = batch['ite'].to(self.device)
                    
                    single_eval_pos = int(len(x) * 0.8)
                    out = self.model(x, a, y, z, single_eval_pos)
                    
                    pi    = out['gmm_pi']      
                    mu    = out['gmm_mu']
                    sigma = out['gmm_sigma']

                    ite_bs1 = ite.permute(1, 0, 2)  
                    loss = self.gmm_nll_loss(pi, mu, sigma, ite_bs1)
                    cate = (pi * mu).sum(dim=-1).unsqueeze(-1)  

                    se = (cate - ite_bs1).pow(2).sum()
                    n = ite_bs1.numel()

                    rmse = torch.sqrt(se / n)
                    sum_sq_err += se.item()
                    total_samples += n

                    total_loss += loss.item()
                    sum_sq_err += se.item()
                    total_samples += n
                    
                    pbar.set_postfix({
                        'loss': loss.item(), 
                    })
        
        val_time = time.time() - val_start_time
        self.validation_times.append(val_time)
        
        avg_loss = total_loss / len(self.val_loader)
        rmse = np.sqrt(sum_sq_err / max(total_samples, 1))
        
        self.val_losses.append(avg_loss)
        self.val_cate_values.append(rmse)
        
        self.scheduler.step(avg_loss)
        
        logger.info(f"Epoch {epoch+1} validation completed in {format_time(val_time)}")
        return avg_loss, rmse
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'training_time': self.total_training_time,
            'epoch_times': self.epoch_times,
        }
        
        checkpoint_path = os.path.join(self.args.save_dir, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model so far, save it separately
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(self.args.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved with validation loss: {val_loss:.4f}")

    def train(self):
        """Main training loop"""
        training_start_time = time.time()
        
        print(f"Starting training on device: {self.device}")
        print(f"Dataset size - Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")
        
        # Make sure save directory exists
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        # Initialize early stopping variables
        early_stop_count = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, accuracy = self.validate(epoch)
            
            epoch_total_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1}/{self.args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch time: {format_time(epoch_total_time)} (Train: {format_time(self.epoch_times[-1])}, Val: {format_time(self.validation_times[-1])})")
            
            # Log to tensorboard
            self.log_metrics(epoch, train_loss, val_loss, accuracy)
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch + 1, val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
                
            if early_stop_count >= self.args.early_stop:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Calculate total training time
        self.total_training_time = time.time() - training_start_time
        
        # Print final timing summary
        self.print_timing_summary()
        
        # Plot training curves
        # self.plot_learning_curves()
        
        # Final checkpoint
        self.save_checkpoint(self.args.epochs, val_loss)
        print("Training complete!")

    def print_timing_summary(self):
        """Print comprehensive timing summary"""
        print("\n" + "="*60)
        print("TRAINING TIME SUMMARY")
        print("="*60)
        
        print(f"Total training time: {format_time(self.total_training_time)}")
        
        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            min_epoch_time = np.min(self.epoch_times)
            max_epoch_time = np.max(self.epoch_times)
            
            print(f"Average epoch time: {format_time(avg_epoch_time)}")
            print(f"Fastest epoch: {format_time(min_epoch_time)}")
            print(f"Slowest epoch: {format_time(max_epoch_time)}")
        
        if self.validation_times:
            avg_val_time = np.mean(self.validation_times)
            print(f"Average validation time: {format_time(avg_val_time)}")
        
        # Estimate time per sample
        total_samples_processed = len(self.train_loader.dataset) * len(self.epoch_times)
        if total_samples_processed > 0:
            time_per_sample = self.total_training_time / total_samples_processed
            print(f"Time per training sample: {time_per_sample*1000:.2f}ms")
        
        # Training efficiency breakdown (if available from first epoch)
        if hasattr(self, 'data_loading_time') and self.data_loading_time > 0:
            total_compute_time = sum(self.epoch_times)
            data_loading_pct = (self.data_loading_time / total_compute_time) * 100
            forward_pct = (self.forward_time / total_compute_time) * 100
            backward_pct = (self.backward_time / total_compute_time) * 100
            
            print(f"\nTime breakdown (estimated from first epoch):")
            print(f"  Data loading: {data_loading_pct:.1f}%")
            print(f"  Forward pass: {forward_pct:.1f}%")
            print(f"  Backward pass: {backward_pct:.1f}%")
            print(f"  Other: {100 - data_loading_pct - forward_pct - backward_pct:.1f}%")
        
        print("="*60)

    def log_metrics(self, epoch, train_loss, val_loss, cate_value):
        """Log metrics to tensorboard"""
        writer = SummaryWriter(log_dir=self.args.log_dir)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('CATE/validation', cate_value, epoch)
        
        # Log timing metrics
        if self.epoch_times:
            writer.add_scalar('Time/epoch_time', self.epoch_times[-1], epoch)
        if self.validation_times:
            writer.add_scalar('Time/validation_time', self.validation_times[-1], epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        writer.close()

    def plot_learning_curves(self):
        """Plot and save learning curves"""
        plt.figure(figsize=(18, 10))
        
        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # CATE values
        plt.subplot(2, 3, 2)
        # Convert CUDA tensor to CPU before plotting
        val_cate_cpu = [x.cpu() if hasattr(x, 'cpu') else x for x in self.val_cate_values]
        plt.plot(val_cate_cpu, label='CATE')
        plt.xlabel('Epoch')
        plt.ylabel('CATE')
        plt.title('CATE')
        plt.legend()
        
        # Epoch times
        plt.subplot(2, 3, 3)
        plt.plot(self.epoch_times, 'g-', label='Epoch Time')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.legend()
        
        # Validation times
        plt.subplot(2, 3, 4)
        plt.plot(self.validation_times, 'r-', label='Validation Time')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Validation Time per Epoch')
        plt.legend()
        
        # Cumulative training time
        plt.subplot(2, 3, 5)
        cumulative_time = np.cumsum(self.epoch_times)
        plt.plot(cumulative_time, 'b-', label='Cumulative Time')
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative Time (seconds)')
        plt.title('Cumulative Training Time')
        plt.legend()
        
        # Training efficiency (time vs loss improvement)
        plt.subplot(2, 3, 6)
        if len(self.train_losses) > 1:
            loss_improvement = [self.train_losses[0] - loss for loss in self.train_losses]
            plt.scatter(cumulative_time, loss_improvement, alpha=0.7)
            plt.xlabel('Cumulative Time (seconds)')
            plt.ylabel('Loss Improvement')
            plt.title('Training Efficiency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, 'learning_curves_with_timing.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Helper function to convert tensor lists to CPU values
        def tensor_list_to_cpu(tensor_list):
            if isinstance(tensor_list, list):
                return [x.cpu().detach().item() if hasattr(x, 'cpu') else x for x in tensor_list]
            elif hasattr(tensor_list, 'cpu'):
                return tensor_list.cpu().detach().numpy().tolist()
            else:
                return tensor_list
        
        # Save metrics to CSV with timing information
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': tensor_list_to_cpu(self.train_losses),
            'val_loss': tensor_list_to_cpu(self.val_losses),
            'cate_value': tensor_list_to_cpu(self.val_cate_values),
            'epoch_time_seconds': self.epoch_times[:len(self.train_losses)],
            'validation_time_seconds': self.validation_times[:len(self.train_losses)],
            'cumulative_time_seconds': np.cumsum(self.epoch_times[:len(self.train_losses)]).tolist()
        })
        
        # Add formatted time columns for readability
        metrics_df['epoch_time_formatted'] = metrics_df['epoch_time_seconds'].apply(format_time)
        metrics_df['cumulative_time_formatted'] = metrics_df['cumulative_time_seconds'].apply(format_time)
        
        metrics_df.to_csv(os.path.join(self.args.save_dir, 'training_metrics_with_timing.csv'), index=False)
        
        # Save timing summary to separate file
        timing_summary = {
            'total_training_time_seconds': self.total_training_time,
            'total_training_time_formatted': format_time(self.total_training_time),
            'average_epoch_time_seconds': np.mean(self.epoch_times) if self.epoch_times else 0,
            'average_epoch_time_formatted': format_time(np.mean(self.epoch_times)) if self.epoch_times else "0s",
            'fastest_epoch_seconds': np.min(self.epoch_times) if self.epoch_times else 0,
            'slowest_epoch_seconds': np.max(self.epoch_times) if self.epoch_times else 0,
            'average_validation_time_seconds': np.mean(self.validation_times) if self.validation_times else 0,
            'total_epochs_completed': len(self.epoch_times),
            'samples_per_second': len(self.train_loader.dataset) / np.mean(self.epoch_times) if self.epoch_times else 0,
        }
        
        with open(os.path.join(self.args.save_dir, 'timing_summary.json'), 'w') as f:
            import json
            json.dump(timing_summary, f, indent=2)

def main():
    """Main entry point for training"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Log some config information
    logger.info(f"Starting training with config:")
    logger.info(f"- Data path: {args.data_path}")
    logger.info(f"- Batch size: {args.batch_size}")
    logger.info(f"- Learning rate: {args.lr}")
    logger.info(f"- Device: {args.device}")
    logger.info(f"- Number of epochs: {args.epochs}")
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    logger.info(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize trainer
    trainer = CATETrainer(args)
    
    # Start training
    try:
        trainer.train()
        
        # Training completed successfully
        end_time = time.time()
        end_datetime = datetime.now()
        total_time = end_time - start_time
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Ended: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {format_time(total_time)}")
        
    except KeyboardInterrupt:
        end_time = time.time()
        interrupted_time = end_time - start_time
        logger.info(f"Training interrupted by user after {format_time(interrupted_time)}")
    except Exception as e:
        end_time = time.time()
        error_time = end_time - start_time
        logger.error(f"Error during training after {format_time(error_time)}: {str(e)}")
        raise
    
    logger.info("Training finished")
    
    # Save final model if not already saved
    if not os.path.exists(os.path.join(args.save_dir, 'final_model.pth')):
        final_training_time = time.time() - start_time
        checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'val_loss': trainer.val_losses[-1] if trainer.val_losses else float('inf'),
            'total_training_time': final_training_time,
            'training_start_time': start_datetime.isoformat(),
            'training_end_time': datetime.now().isoformat(),
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'final_model.pth'))
        logger.info("Final model saved with timing information")
    
   
if __name__ == "__main__":
    main()