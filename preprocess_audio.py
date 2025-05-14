import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

class UrbanSoundDataset(Dataset):
    def __init__(self, dataset, transform=None, target_duration=4.0):
        self.dataset = dataset
        self.transform = transform
        self.target_duration = target_duration
        self.sr = 44100 # Standard sample rate: 22050
        self.target_samples = int(self.target_duration * self.sr)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        audio_array = self.dataset[idx]['audio']['array']
        target_class = self.dataset[idx]['classID']

        # Handle audio that's too short or too long
        if len(audio_array) < self.target_samples:
            #print("Audio array len before padding: ", len(audio_array))
            # Pad with zeros if too short
            audio_array = np.pad(audio_array, (0, self.target_samples - len(audio_array)))
            #print(f"Padded audio array to {self.target_samples} samples")
        else:
            # Truncate if too long
            audio_array = audio_array[:self.target_samples]
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
            sr=self.sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        
        # Convert to tensor
        mel_spec_tensor = torch.FloatTensor(mel_spec_db)
        
        # Encoder expects a 2D tensor with shape [batch size, n_mels, n_ctx]

        return mel_spec_tensor, target_class

def create_dataloader(batch_size, shuffle=True, target_duration=4.0):
    # Load the dataset from Hugging Face
    dataset = load_from_disk("./data/urban8k_ds")
    
    # Convert to pandas for splitting
    df = dataset['train'].to_pandas()
    
    # Get indices for splitting
    train_indices, test_indices = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df['class']
    )
    
    # Create train and test datasets
    train_dataset = UrbanSoundDataset(dataset['train'].select(train_indices), target_duration=target_duration)
    test_dataset = UrbanSoundDataset(dataset['train'].select(test_indices), target_duration=target_duration)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader


if __name__ == "__main__":

    print("Loading the dataset...")
    # Create dataloaders with default parameters
    train_loader, test_loader = create_dataloader()
    
    # Test the dataloader
    for batch, labels in train_loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Label: {labels}")
        break 