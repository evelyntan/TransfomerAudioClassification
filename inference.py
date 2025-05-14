import torch
import torch.nn.functional as F
import wandb
import random
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from preprocess_audio import create_dataloader
from audio_encoder import AudioEncoder

def run_inference():
    # Define class mapping
    vocab = {
        0: "air_conditioner",
        1: "car_horn",
        2: "children_playing",
        3: "dog_bark",
        4: "drilling",
        5: "engine_idling",
        6: "gun_shot",
        7: "jackhammer",
        8: "siren",
        9: "street_music"
    }
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize wandb
    print("Initializing wandb...")
    run = wandb.init(project="audio-classification", job_type="inference")
    
    # Load the best model from local directory
    print("Loading best model from checkpoints...")
    best_model_path = "checkpoints/best_model.pt"
    
    # Initialize model with the same parameters used during training
    model = AudioEncoder(
        n_mels=128,
        n_ctx=341,
        n_state=384,
        n_head=8,
        n_layer=2,
        mlp_dim=768,
        num_classes=10,
        use_classification=True
    ).to(device)
    
    # Load the model weights
    # Set weights_only=False to handle custom classes in the checkpoint
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        # If it's a full checkpoint dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If it's just the state_dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load the test data
    print("Loading test data...")
    _, test_loader = create_dataloader(batch_size=1, shuffle=False)  # Use batch_size=1 for individual samples
    
    # Get a random batch from the test loader
    test_data_iter = iter(test_loader)
    num_samples = len(test_loader)
    random_idx = random.randint(0, num_samples - 1)
    
    # Skip to the random sample
    for _ in range(random_idx):
        next(test_data_iter)
    
    # Get the random sample
    mel_spec, label = next(test_data_iter)
    true_label = label.item()
    
    print(f"Selected random sample (index {random_idx})")
    print(f"True label: {true_label} ({vocab[true_label]})")
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    # Convert PyTorch tensor to numpy for visualization
    mel_spec_np = mel_spec[0].numpy()
    
    # Convert mel spectrogram to log mel spectrogram for visualization
    mel_spec_db = librosa.power_to_db(mel_spec_np, ref=np.max)
    
    # Display the mel spectrogram in an external window
    plt.figure(figsize=(10, 6))
    
    # Use the viridis colormap with enhanced contrast
    librosa.display.specshow(
        mel_spec_db,
        x_axis='time',
        y_axis='hz',
        sr=22050,
        hop_length=512,
        fmax=8000
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram (dB)')
    
    # Add class label at the bottom
    plt.figtext(0.5, 0.01, f"{vocab[true_label]}", ha="center", fontsize=12, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    
    # Save for wandb logging
    plt.savefig('sample_mel.png')
    wandb.log({"sample_mel": wandb.Image('sample_mel.png')})
    
    # Show in external window (non-blocking)
    plt.show(block=False)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Move data to device - original mel_spec tensor for inference
        mel_spec_tensor = mel_spec.to(device)
        
        # Forward pass
        outputs = model(mel_spec_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get the predicted class
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
        confidence = probabilities[0][predicted_label].item() * 100
        
        # Display results
        print("\nInference Results:")
        print(f"True Label: {true_label} ({vocab[true_label]})")
        print(f"Predicted Label: {predicted_label} ({vocab[predicted_label]})")
        print(f"Confidence: {confidence:.2f}%")
        
        # Log top 3 predictions
        print("\nTop 3 predictions:")
        top3_values, top3_indices = torch.topk(probabilities, 3)
        
        top3_results = []
        for i in range(3):
            idx = top3_indices[0][i].item()
            prob = top3_values[0][i].item() * 100
            print(f"{vocab[idx]}: {prob:.2f}%")
            top3_results.append({"class": vocab[idx], "probability": prob})
        
        wandb.log({
            "true_label": vocab[true_label],
            "predicted_label": vocab[predicted_label],
            "confidence": confidence,
            "top3_predictions": top3_results
        })
    
    wandb.finish()
    return {
        "true_label": vocab[true_label],
        "predicted_label": vocab[predicted_label],
        "confidence": confidence
    }

if __name__ == "__main__":
    run_inference()
