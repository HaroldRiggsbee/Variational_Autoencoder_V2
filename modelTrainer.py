import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd

import time

class model_trainer():
  def __init__(self, train_loader, val_loader, epochs, KL_weight, patience, criterion, optimizer, model, save_path, device):
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.epochs = epochs
    self.KL_weight = KL_weight
    self.patience = patience
    self.criterion = criterion
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

  def run_training(self):
    for i in range(self.epochs):
      # Run the training batches
      self.model.train()
      for b, data in enumerate(self.train_loader):
        b+=1

        # Move X_train to the appropriate device
        X_train = data.to(self.device)

        # --- ADD THIS DEBUG BLOCK ---
        if b == 1:  # Only check the first batch of the first epoch
          print(f"DEBUG: X_train device: {X_train.device}")
          print(f"DEBUG: X_train dtype: {X_train.dtype}")
          # Get the device of the first layer of the model
          model_device = next(self.model.parameters()).device
          model_dtype = next(self.model.parameters()).dtype
          print(f"DEBUG: Model device: {model_device}")
          print(f"DEBUG: Model dtype: {model_dtype}")
        # ----------------------------

        # Apply the model
        y_pred = self.model(X_train)

        #caculate MSE_loss
        loss = self.criterion(y_pred, X_train)

        # KL_divergence loss
        KL_loss = self.model.KL_divergence()
        KL_loss = KL_loss.mean()
        self.KL_losses.append(KL_loss.item())

        # Print interim results
        if b%3 == 0:
            print(f'epoch: {i+1:2}  batch: {b:4} [{b* self.train_loader.batch_size:6}/{len(self.train_loader.dataset)}]  loss: {loss.item():10.8f}, KL-loss = {KL_loss:10.8f}')

        #calculate KL_loss
        total_loss =  loss + (self.KL_weight * (i+1)) * KL_loss #added KL annealing so the Network minimizes MSE loss before KL loss
        if i > 9 and KL_loss > 100:
          self.kl_loss_spike = True
          break

        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() #empty cache due to memory issues in Collab


      self.train_losses.append(loss.item())
      print(f'Training loss at Epoch {i+1} is {self.train_losses[-1]}')

      if self.kl_loss_spike:
        print(f"KL spike detected at {self.epochs}")
        torch.save(self.model.state_dict(), self.save_path)
        break
      # Running validation batches
      self.model.eval()
      total_val_loss = 0
      with torch.no_grad():
        for b, X_test in enumerate(self.val_loader):
          X_test = X_test.to(self.device)
          y_val = self.model(X_test)
          val_loss = self.criterion(y_val, X_test)
          KL_loss = self.model.KL_divergence()
          KL_loss = KL_loss.mean()
          current_val_loss =  val_loss + (self.KL_weight * (i+1)) * KL_loss
          total_val_loss += current_bath_loss.item()

      avg_val_loss = total_val_loss / len(self.val_loader)
      self.val_losses.append(avg_val_loss)
      print(f'Validation loss at Epoch {i+1} is {self.val_losses[-1]}')
      elasped_time = time.time() - self.start_time
      print(f"Elasped time is {elasped_time: .0f} seconds.\n")
      #Early stopping check


      if self.train_losses[-1] < self.best_test_loss:
        self.best_test_loss = self.train_losses[-1]
        epochs_without_improvement = 0
      else:
        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
          print(f"Early stopping at epoch {i+1}")
          torch.save(self.model.state_dict(), self.save_path)
          print(f"Model saved to {self.save_path}")
          break

    print(f'\nDuration: {time.time() - self.start_time:.0f} seconds') # print the time elapsed

  #Retrun loss data for graphing
  def return_losses(self):
    return self.train_losses, self.val_losses
