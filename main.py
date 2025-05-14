from train_model import Trainer, Config
from preprocess_audio import create_dataloader
from audio_encoder import AudioEncoder

print("Loading data...")
# 1. Create your data loaders
train_loader, val_loader = create_dataloader(batch_size=32)

print("Data loaded successfully")

print("\nInitializing AudioEncoder...")
# 2. Initialize your model
model = AudioEncoder(
    n_mels=128,
    n_ctx=341,
    n_state=384,
    n_head=8,
    n_layer=2,
    mlp_dim=768,
    num_classes=10,
    use_classification=True
)
print("\nPassing data through encoder...")
# 3. Initialize trainer with default config
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=Config()  # Uses default config values
)

# 4. Start training
trainer.train()