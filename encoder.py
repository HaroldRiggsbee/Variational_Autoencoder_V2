import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
#%matplotlib inline

import time

class ConvolutionalNetwork(nn.Module):
  def __init__(self,input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim, padding, output_padding, device):
    super().__init__()
    self.input_shape = input_shape
    self.conv_filters = conv_filters
    self.conv_kernels = conv_kernels
    self.conv_strides = conv_strides
    self.latent_space_dim = latent_space_dim
    self.padding = padding
    self.output_padding = output_padding
    self.device = device


  #build initial model
  def build_model(self, device):
    self.encoder = self._build_encoder()

    #Create test input to get bottleneck and decoder shapes
    test_input = torch.randn(1, *self.input_shape)
    with torch.no_grad():
        test_output = self.encoder(test_input)
    self.shape_before_bottleneck = test_output.shape
    print(f"Shape before bottleneck is {self.shape_before_bottleneck}")
    self.encoder_output_size_flattened = torch.prod(torch.tensor(self.shape_before_bottleneck)).item()

    #build latent space and decoder
    self.mu, self.log_variance = self._build_latent_space()
    self.decoder = self._build_decoder()

    # Move the model components to the specified device
    self.to(device)


  def _build_encoder(self):
    encoder = nn.Sequential(
        nn.Conv2d(self.input_shape[0], self.conv_filters[0], kernel_size=self.conv_kernels[0], stride=self.conv_strides[0], padding=1),
        nn.LeakyReLU(),
        nn.Dropout(0.05),
        nn.BatchNorm2d(num_features=self.conv_filters[0]),
        nn.Conv2d(self.conv_filters[0], self.conv_filters[1], kernel_size=self.conv_kernels[1], stride=self.conv_strides[1], padding=1),
        nn.LeakyReLU(),
        nn.Dropout(0.05),
        nn.BatchNorm2d(num_features=self.conv_filters[1]),
        nn.Conv2d(self.conv_filters[1], self.conv_filters[2], kernel_size=self.conv_kernels[2], stride=self.conv_strides[2], padding=1),
        nn.LeakyReLU(),
        nn.Dropout(0.05),
        nn.BatchNorm2d(num_features=self.conv_filters[2]),
        nn.Conv2d(self.conv_filters[2], self.conv_filters[3], self.conv_kernels[3], stride=self.conv_strides[3], padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=self.conv_filters[3]),
        nn.Dropout(0.05),
        nn.Conv2d(self.conv_filters[3], self.conv_filters[4], self.conv_kernels[4], stride=self.conv_strides[4], padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=self.conv_filters[4]),
    )
    return encoder

  def _build_decoder(self):
    decoder = torch.nn.Sequential(
        nn.Linear(self.latent_space_dim, self.encoder_output_size_flattened),
        nn.LeakyReLU(),
        nn.Unflatten(1, self.shape_before_bottleneck[1:]),
        nn.ConvTranspose2d(self.conv_filters[4], self.conv_filters[4], kernel_size=self.conv_kernels[4], stride=self.conv_strides[4], padding=self.padding[0], output_padding=self.output_padding[0]),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=self.conv_filters[4]),
        nn.ConvTranspose2d(self.conv_filters[4], self.conv_filters[3], kernel_size=self.conv_kernels[3], stride=self.conv_strides[3], padding=self.padding[1], output_padding=self.output_padding[1]),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=self.conv_filters[3]),
        nn.ConvTranspose2d(self.conv_filters[3], self.conv_filters[2], kernel_size=self.conv_kernels[2], stride=self.conv_strides[2], padding=self.padding[2], output_padding=self.output_padding[2]),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=self.conv_filters[2]),
        nn.ConvTranspose2d(self.conv_filters[2], self.conv_filters[1], kernel_size=self.conv_kernels[1], stride=self.conv_strides[1], padding=self.padding[3], output_padding=self.output_padding[3]),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=self.conv_filters[1]),
        nn.ConvTranspose2d(self.conv_filters[1], 1, kernel_size=(self.conv_kernels[0]), stride=self.conv_strides[0], padding=self.padding[4], output_padding=self.output_padding[4]),
        # nn.Tanh()  # Output in range [-1, 1]
        nn.Sigmoid()
    )
    return decoder

  def _build_latent_space(self):
    mu = nn.Linear(self.encoder_output_size_flattened, self.latent_space_dim)  # Define mu layer
    log_variance = nn.Linear(self.encoder_output_size_flattened, self.latent_space_dim)  # Define logvar layer
    return mu, log_variance

  #Run data through latent space
  def bottleneck(self, X):
    X = torch.flatten(X, start_dim=1)
    self.mu_tensor = self.mu(X)  # Use the defined mu layer
    self.log_variance_tensor = self.log_variance(X)  # Use the defined logvar layer
    epsilon = torch.randn_like(self.log_variance_tensor) #.to(self.log_variance_tensor.device) #Sample point in the latent space
    return self.mu_tensor + torch.exp(0.5 * self.log_variance_tensor) * epsilon

  #calculate KL Loss
  @property
  def KL_divergence(self):
    kl_loss = -0.5 * torch.sum(1 + self.log_variance_tensor - self.mu_tensor.pow(2) - torch.exp(self.log_variance_tensor), dim=1)
    Kl_safeguard = torch.clamp(kl_loss, min=0.5)  # Preventing possible KL posterior collapse
    return Kl_safeguard.to(self.mu_tensor.device)

  #main forward method
  def forward(self, input):
    encoded = self.encoder(input)
    bottleneck = self.bottleneck(encoded)
    decoded = self.decoder(bottleneck)
    return decoded

  #Reconstruction method for testing
  def reconstruct(self, images):
    encoded_image = self.encoder(images)
    latent_representations = self.bottleneck(encoded_image.to(self.device))
    reconstructed_images = self.decoder(latent_representations)
    return reconstructed_images, latent_representations

