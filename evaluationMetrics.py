"""
Evaluation metrics module for audio spectrogram analysis.

This module provides functionality for:
- Selecting random images from a dataset
- Plotting original vs reconstructed spectrograms
- Visualizing latent space representations
- Converting spectrograms back to audio signals
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import display, Audio


def select_images(loader, num_images=4, min_max_values=None):
    """
    Randomly select images from the dataset.

    Args:
        loader: PyTorch DataLoader containing the dataset
        num_images: Number of images to select (default: 4)
        min_max_values: Optional array of min/max normalization values for each sample

    Returns:
        tuple: (sample_images tensor, list of min/max values)
    """
    image_index = np.random.choice(
        len(loader.dataset),
        size=num_images,
        replace=False
    )
    sample_images = []
    sample_min_max = []

    for index in image_index:
        sample_images.append(torch.from_numpy(loader.dataset[index]))
        if min_max_values is not None:
            sample_min_max.append(min_max_values[index])
            print(f"Min/Max values for sample {index}: {sample_min_max[-1]}")

    sample_images = torch.stack(sample_images)
    return sample_images, sample_min_max


def plot_reconstructed_images(images, reconstructed_images, min_max_values, sample_rate, hop_length, scaler=None):
    """
    Plot original and reconstructed spectrograms side by side.

    Args:
        images: Original spectrogram images
        reconstructed_images: Reconstructed spectrogram images
        scaler: Optional scaler object for denormalization
        sample_rate: Audio sample rate (default: 22050)
        hop_length: STFT hop length (default: 512)
    """
    num_images = len(images)
    fig, axes = plt.subplots(
        num_images, 2,
        figsize=(12, 6 * num_images),
        sharex='col'
    )

    # Handle single image case (axes won't be 2D)
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        # Extract spectrogram data
        original_spectrogram = image[0, :, :].cpu().detach().numpy().squeeze()
        reconstructed_spectrogram = reconstructed_image[0, :, :].cpu().detach().numpy().squeeze()

        # Denormalize if scaler is provided
        if scaler is not None:
            original_spectrogram = scaler.inverse_transform(original_spectrogram)
            reconstructed_spectrogram = scaler.inverse_transform(reconstructed_spectrogram)

        # Plot original spectrogram
        librosa.display.specshow(
            original_spectrogram,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis='time',
            y_axis='log',
            cmap="magma",
            ax=axes[i, 0]
        )
        axes[i, 0].set_title(f"Original Spectrogram {i + 1}")

        # Plot reconstructed spectrogram
        librosa.display.specshow(
            reconstructed_spectrogram,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis='time',
            y_axis='log',
            cmap="magma",
            ax=axes[i, 1]
        )
        axes[i, 1].set_title(f"Reconstructed Spectrogram {i + 1}")

    plt.tight_layout()
    plt.show()


def plot_latent_space(latent_representation, sample_labels=None):
    """
    Visualize the 2D latent space representation.

    Args:
        latent_representation: 2D array of latent space coordinates
        sample_labels: Optional labels for coloring points
    """
    plt.figure(figsize=(10, 10))

    scatter_params = {
        'alpha': 0.5,
        's': 2
    }

    if sample_labels is not None:
        scatter_params['c'] = sample_labels
        scatter_params['cmap'] = 'viridis'

    plt.scatter(
        latent_representation[:, 0],
        latent_representation[:, 1],
        **scatter_params
    )

    if sample_labels is not None:
        plt.colorbar()

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.show()


def convert_spectrograms_to_audio(reconstructed_images, scaler, sample_rate, output_dir):
    """Process reconstructed images into audio wave files"""
    if reconstructed_images is not None:
        print("Converting reconstructed spectrograms to audio...")
        for idx, spectrogram in enumerate(reconstructed_images):
            log_spectrogram = spectrogram[0, :, :].cpu().detach().numpy().squeeze()
            denormalized_spectrogram = scaler.inverse_transform(log_spectrogram)
            print(f"Reconstructed {idx + 1} - Max: {denormalized_spectrogram.max():.2f}, "
                  f"Min: {denormalized_spectrogram.min():.2f}")
            spec = librosa.db_to_amplitude(denormalized_spectrogram)
            signal = librosa.istft(spec, hop_length=hop_length)
            #code to save to hard drive
            file_path = os.path.join(output_dir, f"{prefix}_{i}.wav")
            sf.write(file_path, signal, sr)
            print(f"Saved: {file_path}")

