import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPVisionModel, AutoImageProcessor
from matplotlib.colors import LogNorm
import mlflow
import os

# --- Configuration ---
# Set these environment variables before creating MLflow client
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

# Toggle for running experiments
ENABLE_MLFLOW = os.getenv('ENABLE_MLFLOW', 'true').lower() == 'true'
OUTPUT_DIR = Path('./outputs/')
OUTPUT_DIR.mkdir(exist_ok=True) # Ensure output directory exists

# --- MLflow Setup ---
if ENABLE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("feature-visualizations")
    mlflow.start_run()

# --- Model Loading ---
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# --- Helper Functions ---
def get_model_outputs(image, output_attentions=False):
    """Processes an image and returns the model outputs."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=output_attentions)
    return outputs

def visualize_distance_map(image, outputs, save_path):
    """Visualizes the cosine distance from the CLS token to patch tokens."""
    all_tokens = outputs.last_hidden_state
    cls_token = all_tokens[:, 0, :]
    patch_tokens = all_tokens[:, 1:, :]
    
    cosine_distances = 1 - F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)
    
    num_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(num_patches))
    
    distance_map = cosine_distances.reshape(grid_size, grid_size)
    
    viz_size = processor.crop_size["height"]
    resized_image = image.resize((viz_size, viz_size))
    grayscale_image = resized_image.convert('L')
    
    distance_map_resized = F.interpolate(
        distance_map.unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(grayscale_image, cmap='gray')
    heatmap = ax.imshow(distance_map_resized.numpy(), norm=LogNorm(), cmap='hot', alpha=0.5)
    fig.colorbar(heatmap, ax=ax)
    ax.axis('off')
    ax.set_title("Outlier Patch Visualization (Brighter = More Unique)")
    plt.savefig(save_path)
    plt.close(fig)

def visualize_attention_map(image, outputs, save_path):
    """Visualizes the attention from the CLS token to the patch tokens."""
    attention_weights = outputs.attentions[-1]
    avg_attention = attention_weights.mean(dim=1)
    cls_to_patch_attention = avg_attention[0, 0, 1:]
    
    num_patches = cls_to_patch_attention.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    attention_map = cls_to_patch_attention.reshape(grid_size, grid_size)
    
    viz_size = processor.crop_size["height"]
    resized_image = image.resize((viz_size, viz_size))
    grayscale_image = resized_image.convert('L')
    
    attention_map_resized = F.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(grayscale_image, cmap='gray')
    heatmap = ax.imshow(attention_map_resized.numpy(), cmap='hot', alpha=0.5)
    fig.colorbar(heatmap, ax=ax)
    ax.axis('off')
    ax.set_title("CLS Token Attention (Brighter = More Contribution)")
    plt.savefig(save_path)
    plt.close(fig)

# --- Main Processing Loop ---
data_path = Path(__file__).parent.parent / 'yellow_car'
image_fns = list(data_path.glob('*.jpg'))
image_fns = image_fns[:10]
print(f"Found {len(image_fns)} images to process")

for i, image_fn in enumerate(image_fns):
    print(f"Processing {i+1}/{len(image_fns)}: {image_fn.name}")
    
    image = Image.open(image_fn)
    
    # Get model outputs, including attentions
    outputs = get_model_outputs(image, output_attentions=True)
    
    # Generate and save the distance visualization
    distance_save_path = OUTPUT_DIR / f'distance-vis_{i}.png'
    visualize_distance_map(image, outputs, distance_save_path)
    
    # Generate and save the attention visualization
    attention_save_path = OUTPUT_DIR / f'attention-vis_{i}.png'
    visualize_attention_map(image, outputs, attention_save_path)
    
    # Log artifacts to MLflow if enabled
    if ENABLE_MLFLOW:
        mlflow.log_artifact(distance_save_path)
        mlflow.log_artifact(attention_save_path)

if ENABLE_MLFLOW:
    mlflow.end_run()