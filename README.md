# Audio Classification with Deep Learning

This repository contains an audio classification system using deep learning to classify urban sounds. The model uses a transformer-based architecture to identify different urban sound categories from audio spectrograms.

## Project Overview

This project implements a deep learning-based audio classification system for the UrbanSound8K dataset. The system:

- Processes audio files into mel spectrogram representations
- Builds a transformer-based encoder architecture for audio classification
- Trains on urban sound categories (car horns, dog barks, sirens, etc.)
- Provides tools for inference and visualization of results

## Features

- **Audio preprocessing pipeline**: Converts raw audio to mel spectrograms
- **Transformer-based architecture**: Uses self-attention for audio feature extraction
- **Training framework**: Complete training loop with validation
- **Visualization tools**: Display spectrograms and model predictions
- **Weights & Biases integration**: Track experiments and visualize results
- **Inference script**: Run predictions on new audio samples

## Dataset

This project uses the UrbanSound8K dataset which contains 8732 labeled sound excerpts (≤4s) of urban sounds from 10 classes:
- Air conditioner
- Car horn
- Children playing
- Dog bark
- Drilling
- Engine idling
- Gun shot
- Jackhammer
- Siren
- Street music

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
```bash
# The dataset will be downloaded when running the preprocessing script
```

## Project Structure

```
.
├── audio_encoder.py        # Model architecture definition
├── preprocess_audio.py     # Audio preprocessing utilities
├── train_model.py          # Training loops and utilities
├── main.py                 # Main script to train the model
├── inference.py            # Inference script for model testing
├── requirements.txt        # Project dependencies
├── data/                   # Data directory
│   └── urban8k_ds/         # Processed dataset
├── checkpoints/            # Saved model checkpoints
└── README.md               # Project documentation
```

## Usage

### Data Preprocessing

The preprocessing script converts audio files to mel spectrograms:

```bash
# The dataset is loaded and processed automatically when running the training script
```

### Training

Train the model using:

```bash
python main.py
```

This will:
1. Load and preprocess the audio data
2. Initialize the audio encoder model
3. Train for the specified number of epochs
4. Save model checkpoints
5. Log metrics to Weights & Biases

### Inference

Run inference on test samples:

```bash
python inference.py
```

This will:
1. Load a trained model from the checkpoints directory
2. Select a random test sample
3. Display the mel spectrogram
4. Show prediction results with confidence scores

## Model Architecture

The audio encoder model consists of:

- Convolutional layers for processing mel spectrograms
- Transformer encoder blocks with multi-head self-attention
- Classification head for predicting audio classes

## Results

The model achieves over 82% validation accuracy on the UrbanSound8K dataset. Detailed metrics and visualizations are available in the Weights & Biases dashboard.

## Dependencies

Main dependencies include:
- torch
- librosa
- soundfile
- numpy
- matplotlib
- wandb
- tqdm
- scikit-learn
- datasets

## Acknowledgments

- UrbanSound8K dataset creators: https://huggingface.co/datasets/danavery/urbansound8K
