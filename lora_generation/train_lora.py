#!/usr/bin/env python3
"""
Simple LoRA Training Script for MacBook
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer
from tqdm import tqdm
import glob
import numpy as np

# Configuration
IMAGE_FOLDER = "./dataset"  # Change this to your image folder
OUTPUT_DIR = "./output"
CAPTION = "photo of a wife123"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MAX_TRAIN_STEPS = 2000
LEARNING_RATE = 1e-4
RESOLUTION = 512
SAVE_EVERY = 100

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device (Apple Silicon)")
else:
    device = torch.device("cpu")
    print(f"Using CPU (slow)")

# Load model
print(f"Loading model: {BASE_MODEL}")
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL)
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.to(device)
vae = pipe.vae.to(device)
unet = pipe.unet.to(device)
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# Freeze all parameters
for param in unet.parameters():
    param.requires_grad = False

# Unfreeze only the attention layers for LoRA training
trainable_layers = []
for name, module in unet.named_modules():
    if isinstance(module, torch.nn.Linear) and any(
        part in name for part in ["to_q", "to_k", "to_v", "to_out.0"]
    ):
        for param in module.parameters():
            param.requires_grad = True
            trainable_layers.append(name)

print(f"Trainable layers: {len(trainable_layers)}")
trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Set up optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()), lr=LEARNING_RATE
)


# Load dataset
def load_images(image_dir, resolution):
    extensions = ["jpg", "jpeg", "png", "webp"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
        images.extend(glob.glob(os.path.join(image_dir, f"*.{ext.upper()}")))

    print(f"Found {len(images)} images")

    processed_images = []
    for img_path in images:
        try:
            img = Image.open(img_path).convert(
                "RGB").resize((resolution, resolution))
            processed_images.append(img)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return processed_images


# Load images
train_images = load_images(IMAGE_FOLDER, RESOLUTION)
if len(train_images) == 0:
    raise ValueError(f"No valid images found in {IMAGE_FOLDER}")

# Training loop
print("Starting training...")
text_encoder.eval()
vae.eval()
unet.train()

global_step = 0
progress_bar = tqdm(range(MAX_TRAIN_STEPS))

while global_step < MAX_TRAIN_STEPS:
    # Random batch of 1 image
    img_idx = torch.randint(0, len(train_images), (1,)).item()
    image = train_images[img_idx]

    # Convert image to latents
    pixel_values = (
        torch.from_numpy(np.array(image) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    with torch.no_grad():
        # Encode text
        text_inputs = tokenizer(
            [CAPTION],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_embeddings = text_encoder(text_inputs.input_ids)[0]

        # Encode images
        latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

    # Prepare noise
    noise = torch.randn_like(latents)
    timestep = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (1,), device=device
    )

    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)

    # Predict noise
    model_pred = unet(noisy_latents, timestep, text_embeddings).sample

    # Calculate loss
    loss = F.mse_loss(model_pred, noise, reduction="mean")

    # Backpropagate and optimize
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Update progress
    progress_bar.update(1)
    progress_bar.set_description(f"Loss: {loss.item():.4f}")
    global_step += 1

    # Save checkpoint
    if global_step % SAVE_EVERY == 0 or global_step == MAX_TRAIN_STEPS:
        # Extract trainable weights
        state_dict = {}
        for name, param in unet.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.cpu().clone()

        # Save to disk
        save_path = os.path.join(OUTPUT_DIR, f"lora-{global_step}.pt")
        torch.save(state_dict, save_path)
        print(f"Saved checkpoint to {save_path}")

print("Training complete!")

# Save model metadata
meta_path = os.path.join(OUTPUT_DIR, "lora_metadata.txt")
with open(meta_path, "w") as f:
    f.write(f"Base model: {BASE_MODEL}\n")
    f.write(f"Caption: {CAPTION}\n")
    f.write(f"Training steps: {MAX_TRAIN_STEPS}\n")
    f.write(f"Resolution: {RESOLUTION}\n")
    f.write(f"Trainable layers: {len(trainable_layers)}\n")
    f.write(f"Trainable parameters: {trainable_params}\n")
    f.write(f"Device: {device}\n")

print(f"Saved metadata to {meta_path}")
