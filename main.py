#import packages



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import librosa
import librosa.display
import os
import pickle
from IPython.display import Audio, display
import time


from encoder import ConvolutionalNetwork
from modelTrainer import model_trainer

from preprocessing import PreprocessingPipeline
from evaluationMetrics import select_images, plot_reconstructed_images, convert_spectrograms_to_audio

#from results import DataDisplay
#!pip install auraloss==0.3.0
#from torch.nn import MultiResolutionSTFTLoss
from IPython.display import Audio


#memory management when using limited GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

#import preprocessing
frame_size = 4096
hop_length = 1024
duration = 0.9  # in seconds
sample_rate = 44100  # SR for kicks
mono = True

train_split = 0.80
val_split = 0.10


spectrograms_save_dir = "spectrograms"
file_dir = "samples"
audio_save_dir = "savedaudio"
save_path= "savedmodels/my_model.pth"

#initialize model parameters
torch.manual_seed(42)
conv_filters=(512, 512, 256, 128, 64)
conv_kernels=(3, 3, 3, 3, 3)
conv_strides=(2, 2, 2, 2, (2,1))
padding = (1,1,1,1,1)
output_padding = ((1,0),(1,0),(1,1),(1,1),(1,0)) #padding for decoder output to match input.
latent_space_dim= 2048
epochs = 100
KL_weight = .0008
patience = 15 #Patience level for early stopping

# preprocessing_pipeline = PreprocessingPipeline(spectrograms_save_dir, file_dir, sample_rate, duration, mono, frame_size,
#                                                hop_length)
# preprocessing_pipeline.process(file_dir)

def load_spectrograms(spectrograms_path):
  _data_pool = []
  file_paths = []
  for root, _, file_names in os.walk(spectrograms_save_dir):
    for file_name in file_names:
      file_path = os.path.join(root, file_name)
      spectrogram = np.load(file_path)
      file_paths.append(file_path)
      _data_pool.append(spectrogram)
  _data_pool = np.array(_data_pool)
  #print(x_train.shape)
  _data_pool = _data_pool[:, np.newaxis, ...] #creating a new axis for channel
  print(f"data pool Shape:{_data_pool.shape}")
  return _data_pool, file_paths

def normalizer(_data_pool):
    _original_spectrogram_values = []
    for values in _data_pool:
        min_val = values.min()
        max_val = values.max()
        _original_spectrogram_values.append((min_val, max_val))
    #normalize processed data
    global scaler
    scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
    # Normalize along the frequency and time dimensions (axes 2 and 3)
    num_samples = _data_pool.shape[0]
    _normalized_data_pool = np.zeros_like(_data_pool, dtype=np.float32)
    for i in range(num_samples):
        _normalized_data_pool[i, 0] = scaler.fit_transform(_data_pool[i, 0])
        # Applying fit_transform to each sample individually
    print(_normalized_data_pool.max())
    print(_normalized_data_pool.min())
    print(f"Shape: {_normalized_data_pool.shape}")
    return _normalized_data_pool, _original_spectrogram_values


#load data into tensors
def loader(_normalized_data_pool, _original_spectrogram_values):
    #Prestort training, validation, and testing batches
    train_end = int(len(_original_spectrogram_values) * train_split)
    val_end = train_end + int(len(_original_spectrogram_values) * val_split)

    #Global declaration of batch lengths. This is so it can be used later when doing evaluations.
    global train_min_max, val_min_max, test_min_max
    train_min_max = _original_spectrogram_values[:train_end]
    val_min_max = _original_spectrogram_values[train_end:val_end]
    test_min_max = _original_spectrogram_values[val_end:]

    #Convert data to Tensor
    data_tensor = torch.tensor(_normalized_data_pool).float()
    #Loading data into training, validation, and testing subsets
    train = DataLoader(torch.utils.data.Subset(data_tensor, range(0, train_end)), batch_size=25, shuffle=True)
    val = DataLoader(torch.utils.data.Subset(data_tensor, range(train_end, val_end)), batch_size=15, shuffle=False)
    test = DataLoader(torch.utils.data.Subset(data_tensor, range(val_end,len(_original_spectrogram_values ))), batch_size=4, shuffle=False)
    #test_min_max_values = _original_spectrogram_values[57:69]
    return train, val, test

def get_device():
    """switches between GPU devices"""
    if torch.cuda.is_available():
        return "cuda"       # For Google Colab
    elif torch.backends.mps.is_available():
        return "mps"        # For your MacBook AMD GPU
    else:
        return "cpu"        # Fallback


def prepare_environment():
    """Handles data loading and returns everything needed for the model."""
    data_pool, _ = load_spectrograms(spectrograms_save_dir)

    if data_pool.size == 0:
        raise FileNotFoundError("No spectrograms found.")

    # These stay local to this function until returned
    normalized_data = normalizer(data_pool)
    in_shape = normalized_data.shape[1:]
    train_dl, val_dl, test_dl = loader(normalized_data)

    # We return them as a tuple
    return normalized_data, in_shape, train_dl, val_dl, test_dl

def run_main():
    #Main switch between training and EVAL modes.
    try:
        user_input = input(f"Options: 1. train  2. load previous model")
        choice = int(user_input)
    except ValueError:
        print(f"Invalid input: '{user_input}' is not a number. Please enter 1 or 2.")
        return

    #Load data into loaders
    data_pool, file_paths = load_spectrograms(spectrograms_save_dir)
    # print(f"Data pool max/min {data_pool.max(), data_pool.min()}")
    if data_pool.size == 0:
        print("Error: data_pool is empty. Check your data loading logic!")
        # Stop the script or handle the error
    else:
        print(f"Data pool max/min {data_pool.max()}, {data_pool.min()}")

    normalized_data_pool, original_spectrogram_values = normalizer(data_pool)
    input_shape = normalized_data_pool.shape[1:]
    train_loader, val_loader, test_loader = loader(normalized_data_pool, original_spectrogram_values)
    device = get_device()
    print(f"Running on: {device}")
    print(f"Current input shape is {input_shape}")

    #Initiate Training mode
    if choice == 1:
        new_data = input("Do you want to scan new data? (y/n): ")
        if new_data.lower() == 'y':
            #reads sample folder and creates new spectrograms (repeated files will be rewritten)
            preprocessing_pipeline = PreprocessingPipeline(spectrograms_save_dir, file_dir, sample_rate, duration,
                                                           mono, frame_size,
                                                           hop_length)
            preprocessing_pipeline.process(file_dir)


        model = ConvolutionalNetwork(input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim,
                                     padding, output_padding, device)
        model.build_model(device=device)
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

        # run model summary to check parameters
        summary(model, input_size=(1, *input_shape), device=device)
        train_model = model_trainer(train_loader, val_loader, epochs, KL_weight, patience, criterion, optimizer, model,
                                    save_path, device)
        train_model.run_training()
        print("Model training is finished!")
        run_main()


    #Initiate EVAL mode
    elif choice == 2:
        print("Loading model")
        eval_model = ConvolutionalNetwork(input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim,
                                          padding, output_padding, device)
        eval_model.build_model(device=device)
        eval_model.load_state_dict(torch.load(save_path))
        eval_model.to(device)
        eval_model.eval()
        try:
            numSampleInput = input("How many samples do you want to save?")
            numSamples = int(numSampleInput)
        except ValueError:
            print(f"Invalid input: '{numSampleInput}' is not a number. Please enter 1 or 2.")
            return
        #1model.eval()
        # train_min_max_values = original_spectrogram_values[:375]
        test_min_max_values = original_spectrogram_values[:len(train_loader.dataset)]
        sample_images, sample_min_max = select_images(test_loader, numSamples, min_max_values=test_min_max)
        sample_images = sample_images.to(device)
        reconstructed_images, _ = eval_model.reconstruct(sample_images)
        plot_reconstructed_images(sample_images.cpu(), reconstructed_images.cpu(), sample_min_max, sample_rate, hop_length, scaler=scaler)
        save_input = input(f"Do you want to save the reconstructed Waveforms(y/n)?")
        if save_input == 'y':
            convert_spectrograms_to_audio(reconstructed_images, scaler, sample_rate, audio_save_dir)


    #else: raise valueerror("please choose one or two")

if __name__ == "__main__":
    run_main()