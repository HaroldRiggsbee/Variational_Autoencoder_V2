#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
#%matplotlib inline

import librosa
import librosa.display
import os
import pickle

import time

class PreprocessingPipeline:
  def __init__(self, SPECTROGRAMS_SAVE_DIR, FILES_DIR, SAMPLE_RATE, DURATION, MONO, FRAME_SIZE, HOP_LENGTH):
    self.file_path = FILES_DIR
    self.mono = MONO
    self.save_path = SPECTROGRAMS_SAVE_DIR
    self.sample_rate = SAMPLE_RATE
    self.duration = DURATION
    self.frame_size =  FRAME_SIZE
    self.hop_length = HOP_LENGTH
    self.num_expected_sample = int(self.sample_rate * self.duration)

  def _file_load(self, file_path):
    signal = librosa.load(file_path, sr = self.sample_rate, duration=self.duration, mono= self.mono)[0]
    return signal

  def _apply_padding(self, signal):
    num_missing_samples = self.num_expected_sample - len(signal)
    print(f"number of missing samples {num_missing_samples}")
    padded_signal = np.pad(signal, (0, num_missing_samples),mode="constant")
    print(f"new padded shape {padded_signal.shape}")
    return padded_signal

  def _extract_harmonics(self, signal):
    harmonic = librosa.effects.harmonic(signal, margin=4.0)
    #_harmonic, _transient = librosa.effects.hpss(signal, margin=(3.0, 7.0))
    return harmonic

  def _augment_audio(self, signal):
    n_steps = np.random.randint(-1, 1)
    augmented_audio = librosa.effects.pitch_shift(signal, sr=self.sample_rate, n_steps=n_steps)
    return augmented_audio

  def _extract(self,signal):
    #n_fft = min(self.frame_size, len(signal))
    stft = librosa.stft(signal,n_fft=self.frame_size,hop_length=self.hop_length)[:-1]
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram

  def _save_feature(self, feature, file_path, count):
    save_path = self._generate_save_path(file_path, count)
    np.save(save_path, feature)

  def _save(self, data, save_path):
    with open(save_path, "wb") as f:
      pickle.dump(data, f)

  def _generate_save_path(self,file_path, count):
    file_name = os.path.split(file_path)[1]
    save_path = os.path.join(self.save_path, file_name + ".npy")
    print(f"Saved: {self.save_path, file_name + str(count)}.npy")
    return save_path

  def process(self,audio_file_dir):
    file_count = 0
    for root, _, files in os.walk(audio_file_dir):
      for file in files:
        #checks for DS_store on macOS
        if file.endswith(".wav") or file.endswith(".mpg"):
          file_path = os.path.join(root, file)
          self._process_file(file_path)
        else: continue
        # if file_count < 5:
        #   signal = self.file_load(file_path)
        #   plt.figure(figsize=(20,7))
        #   librosa.display.waveshow(signal, sr = self.sample_rate)
        #   plt.title(f"Waveform of {file}")
        #   plt.show()
        #   file_count += 1
        #print(f"Processed file {file_path}")
    #self.saver.save_min_max_values(self.min_max_values)
    print("Files complete")

  def _process_file(self, file_path):
    signal = self._file_load(file_path)
    print(f"Shape before padding {signal.shape}")
    if len(signal) < self.num_expected_sample:
      signal = self._apply_padding(signal)
    else: print("No padding needed")
    pitch_count = 0
    # while pitch_count < 2:
    #   pitched_audio = self.augment_audio(signal)
    harmonic_signal = self._extract_harmonics(signal)
    feature = self._extract(harmonic_signal)
    self._save_feature(feature, file_path, pitch_count)
    #   pitch_count += 1