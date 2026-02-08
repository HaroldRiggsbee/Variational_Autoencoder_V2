import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd

import time


class model_trainer():
    def __init__(self, train_loader, val_loader, epochs, KL_weight, patience, optimizer, model, save_path, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.KL_weight = KL_weight
        self.patience = patience
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.KL_losses = []
        self.train_correct = []
        self.test_correct = []
        self.best_test_loss = float("inf")
        self.epochs_without_improvement = 0
        self.model = model
        self.start_time = time.time()
        self.save_path = save_path
        self.kl_loss_spike = False
        self.device = device

    def compute_loss(self, y_pred, X_train):
        """Initial MSE loss switched out to weighted loss to account for sparse regions in the spectrograms.
        data points with audio are given a 20x weight."""
        # recon_loss = F.mse_loss(y_pred, X_train)

        weight_mask = torch.where(X_train > 0.05, 20.0, 1.0)
        weighted_mse = torch.mean(weight_mask * (y_pred - X_train) ** 2)
        return weighted_mse

        # combined reconstruction loss
        #recon_loss = self.linear_weight * linear_loss + self.log_weight * log_loss

        #return recon_loss

    def run_training(self):
        """Runs training loop and prints out losses and NN status."""
        self.model.train()
        for epoch in range(self.epochs):
            # Run the training batches
            # declare variables for average loss calculation
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0

            for b, data in enumerate(self.train_loader):
                # In run_training, after the first batch of first epoch:
                b += 1
                num_batches += 1

                # Move X_train to the appropriate device
                X_train = data.to(self.device)

                # Apply the model
                y_pred = self.model(X_train)

                recon_loss = self.compute_loss(y_pred, X_train)

                KL_loss = self.model.KL_divergence
                KL_loss = KL_loss.mean()

                current_kl_weight = self.KL_weight * min(1.0, (epoch + 1) / 50)
                total_loss = recon_loss + current_kl_weight * KL_loss
                epoch_kl_loss += KL_loss.item()

                # Update parameters
                self.optimizer.zero_grad()
                total_loss.backward()

                # Add gradient clipping for stability
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                torch.cuda.empty_cache()  # empty cache due to memory issues in Collab

                #add loss items to list
                epoch_recon_loss += recon_loss.item()
                epoch_total_loss += total_loss.item()

                # Print interim results
                if b % 3 == 0:
                    print(f'epoch: {epoch + 1:2}  batch: {b:4} '
                          f'[{b * self.train_loader.batch_size:6}/{len(self.train_loader.dataset)}]  '
                          f'recon_loss: {recon_loss.item():10.8f}, '
                          f'KL-loss: {KL_loss.item():10.8f}')

            # Calculate average losses for the epoch
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches

            # Store losses
            self.train_losses.append(avg_total_loss)
            self.KL_losses.append(avg_kl_loss)

            print(f'\nEpoch {epoch + 1} Summary:')
            print(f'  Reconstruction loss: {avg_recon_loss:.6f}')
            print(f'  KL loss: {avg_kl_loss:.6f}')
            print(f'  Total loss: {avg_total_loss:.6f}')

            # check validation loss
            if self.val_loader is not None:
                # FIX: Use 'epoch' instead of 'i'
                val_loss = self.validate(epoch)
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < self.best_test_loss:
                    self.best_test_loss = val_loss
                    self.epochs_without_improvement = 0
                    # Save best model
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f'  ✓ Best model saved with validation loss: {val_loss:.6f}')
                else:
                    self.epochs_without_improvement += 1
                    print(f'  No improvement for {self.epochs_without_improvement} epochs')

                if self.epochs_without_improvement >= self.patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

        # Print training summary
        elapsed_time = time.time() - self.start_time
        print(f'\nTraining completed in {elapsed_time / 60:.2f} minutes')
        print(f'Best validation loss: {self.best_test_loss:.6f}')

    # Running validation batches
    def validate(self, epoch):
        self.model.eval()

        val_recon_loss = 0.0
        val_kl_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for b, X_val in enumerate(self.val_loader):
                num_batches += 1
                X_val = X_val.to(self.device)
                y_val = self.model(X_val)

                # Linear loss
                recon_loss = self.compute_loss(y_val, X_val)

                KL_loss = self.model.KL_divergence
                KL_loss = KL_loss.mean()

                # if epoch < 10:
                #     total_val_loss = recon_loss  # NO KL
                # else:
                current_kl_weight = self.KL_weight * min(1.0, (epoch + 1) / 50)
                total_val_loss = recon_loss + current_kl_weight * KL_loss

                val_recon_loss += recon_loss.item()
                val_kl_loss += KL_loss.item()

            # Calculate averages
            avg_recon = val_recon_loss / num_batches
            avg_kl = val_kl_loss / num_batches
            print(f'  Validation - Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}, Total: {total_val_loss:.6f}')


            return total_val_loss

    # Retrun loss data for graphing
    def return_losses(self):
        return self.train_losses, self.val_losses