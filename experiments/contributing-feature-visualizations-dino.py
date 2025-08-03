import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import Dinov2Model, AutoImageProcessor # Changed from CLIPVisionModel
from matplotlib.colors import LogNorm
import mlflow
import os
from scipy.ndimage import binary_closing

# --- Configuration ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

ENABLE_MLFLOW = os.getenv('ENABLE_MLFLOW', 'true').lower() == 'true'
OUTPUT_DIR = Path(__file__).parent / 'outputs_dino' # Changed output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# --- MLflow Setup ---
if ENABLE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("dino-feature-visualizations") # New experiment name
    mlflow.start_run()

# --- Model Loading ---
# Changed model to DINOv2
print("Loading DINOv2 model...")
model = Dinov2Model.from_pretrained("facebook/dinov2-base")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
print("Model loaded.")

# --- Helper Functions (No changes needed) ---
def get_model_outputs(image, output_attentions=False):
    """Processes an image and returns the model outputs."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=output_attentions)
    return outputs

def get_tokens(outputs):
    all_tokens = outputs.last_hidden_state
    # NOTE: For DINOv2, the CLS token is still at index 0
    cls_token = all_tokens[:, 0, :]
    patch_tokens = all_tokens[:, 1:, :]
    return all_tokens, cls_token, patch_tokens

def get_cosine_distances(cls_token, patch_tokens):
    return 1 - F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)

def visualize_mask(image, outputs, save_path):
    """Visualizes the binary foreground mask generated from attention."""
    attention_weights = outputs.attentions[-1].mean(dim=1)
    cls_to_patch_attention = attention_weights[0, 0, 1:]
    
    attention_threshold = torch.quantile(cls_to_patch_attention, 0.80)
    foreground_mask = cls_to_patch_attention > attention_threshold
    
    num_patches = cls_to_patch_attention.shape[0]
    grid_size = int(np.sqrt(num_patches))
    mask_2d = foreground_mask.reshape(grid_size, grid_size)
    
    # --- Start of The Fix ---
    # Apply morphological closing to fill holes and connect components
    # Note: scipy works on NumPy arrays, so we convert the tensor
    cleaned_mask_2d = binary_closing(mask_2d.numpy())
    # --- End of The Fix ---

    viz_size = processor.crop_size["height"]
    
    # Use the cleaned mask for visualization
    mask_resized = F.interpolate(
        torch.from_numpy(cleaned_mask_2d).float().unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='nearest'
    ).squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(mask_resized.numpy(), cmap='gray')
    ax.axis('off')
    ax.set_title("Cleaned Foreground Mask (White = Foreground)")
    plt.savefig(save_path)
    plt.close(fig)

def visualize_distance_map(image, outputs, save_path):
    """Visualizes the cosine distance from the CLS token to patch tokens."""
    _, cls_token, patch_tokens = get_tokens(outputs)
    cosine_distances = get_cosine_distances(cls_token, patch_tokens)

    # 1. Create the attention map that will serve as our mask
    attention_weights = outputs.attentions[-1].mean(dim=1) # Average heads
    cls_to_patch_attention = attention_weights[0, 0, 1:] # Shape: [196]

    # 2. Create a boolean foreground mask by setting a threshold
    # You can tune this threshold. A good starting point is the mean attention.
    attention_threshold = cls_to_patch_attention.mean()
    foreground_mask = cls_to_patch_attention > attention_threshold # Shape: [196]

    # 3. Apply the mask to your distance scores
    # We'll create a copy and set background distances to a very low value (-1)
    # so they are never chosen by `torch.topk`.
    masked_distances = cosine_distances.clone()
    # Ensure mask has the same shape as cosine_distances
    foreground_mask_expanded = foreground_mask.expand_as(cosine_distances)
    masked_distances[~foreground_mask_expanded] = -1.0 # Invert mask to select background

    # 4. Find the top outlier patches from the FOREGROUND only
    num_outlier_patches = int(patch_tokens.shape[1] * 0.10)
    _, top_indices = torch.topk(masked_distances, k=num_outlier_patches)

    # 5. Create the uniqueness vector (this part is the same as before)
    outlier_embeddings = patch_tokens.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
    uniqueness_vector = torch.mean(outlier_embeddings, dim=1)

    # 6. Concatenate for the final, background-filtered vector
    final_feature_vector = torch.cat([cls_token, uniqueness_vector], dim=1)
    
    num_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(num_patches))
    
    # Create the distance map with foreground masking applied
    distance_map = cosine_distances.reshape(grid_size, grid_size)
    foreground_mask_2d = foreground_mask.reshape(grid_size, grid_size)
    
    # Apply foreground mask to the visualization
    masked_distance_map = distance_map.clone()
    masked_distance_map[~foreground_mask_2d] = 0.0  # Set background to 0 for visualization
    
    # DINOv2 base model uses 224x224 images
    viz_size = processor.crop_size["height"]
    resized_image = image.resize((viz_size, viz_size))
    grayscale_image = resized_image.convert('L')
    
    # Resize the masked distance map
    masked_distance_map_resized = F.interpolate(
        masked_distance_map.unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # Create a mask for the resized visualization
    foreground_mask_resized = F.interpolate(
        foreground_mask_2d.float().unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='nearest'
    ).squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(grayscale_image, cmap='gray')
    
    # Apply the mask to the heatmap - only show foreground regions
    masked_heatmap_data = masked_distance_map_resized.numpy()
    masked_heatmap_data[foreground_mask_resized.numpy() == 0] = np.nan  # Hide background
    
    heatmap = ax.imshow(masked_heatmap_data, norm=LogNorm(), cmap='rainbow', alpha=0.7)
    fig.colorbar(heatmap, ax=ax)
    ax.axis('off')
    ax.set_title("DINOv2 Foreground Outlier Patches (Brighter = More Unique)")
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
    
    # DINOv2 base model uses 224x224 images
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
    # ax.imshow(grayscale_image, cmap='gray')
    heatmap = ax.imshow(attention_map_resized.numpy(), cmap='rainbow', norm=LogNorm(), alpha=0.5)
    fig.colorbar(heatmap, ax=ax)
    ax.axis('off')
    ax.set_title("DINOv2 CLS Token Attention (Brighter = More Contribution)")
    plt.savefig(save_path)
    plt.close(fig)

def visualize_comparison(image, outputs, save_path):
    """Visualizes both original and foreground-masked distance maps side by side."""
    _, cls_token, patch_tokens = get_tokens(outputs)
    cosine_distances = get_cosine_distances(cls_token, patch_tokens)
    
    # Get attention for foreground masking
    attention_weights = outputs.attentions[-1].mean(dim=1)
    cls_to_patch_attention = attention_weights[0, 0, 1:]
    attention_threshold = cls_to_patch_attention.mean()
    foreground_mask = cls_to_patch_attention > attention_threshold
    
    num_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(num_patches))
    
    # Original distance map
    distance_map = cosine_distances.reshape(grid_size, grid_size)
    
    # Foreground-masked distance map
    foreground_mask_2d = foreground_mask.reshape(grid_size, grid_size)
    masked_distance_map = distance_map.clone()
    masked_distance_map[~foreground_mask_2d] = 0.0
    
    # DINOv2 base model uses 224x224 images
    viz_size = 224
    resized_image = image.resize((viz_size, viz_size))
    grayscale_image = resized_image.convert('L')
    
    # Resize both maps
    distance_map_resized = F.interpolate(
        distance_map.unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    masked_distance_map_resized = F.interpolate(
        masked_distance_map.unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # Create mask for resized visualization
    foreground_mask_resized = F.interpolate(
        foreground_mask_2d.float().unsqueeze(0).unsqueeze(0),
        size=(viz_size, viz_size),
        mode='nearest'
    ).squeeze()
    
    # Apply mask to hide background
    masked_heatmap_data = masked_distance_map_resized.numpy()
    masked_heatmap_data[foreground_mask_resized.numpy() == 0] = np.nan
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original visualization
    ax1.imshow(grayscale_image, cmap='gray')
    heatmap1 = ax1.imshow(distance_map_resized.numpy(), norm=LogNorm(), cmap='rainbow', alpha=0.7)
    fig.colorbar(heatmap1, ax=ax1)
    ax1.axis('off')
    ax1.set_title("Original: All Patches")
    
    # Foreground-masked visualization
    ax2.imshow(grayscale_image, cmap='gray')
    heatmap2 = ax2.imshow(masked_heatmap_data, norm=LogNorm(), cmap='rainbow', alpha=0.7)
    fig.colorbar(heatmap2, ax=ax2)
    ax2.axis('off')
    ax2.set_title("Foreground-Masked: Only Foreground Patches")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# --- Main Processing Loop ---
# Using a relative path for robustness
data_path = Path(__file__).parent.parent / 'yellow_car'
image_fns = list(data_path.glob('*.jpg'))
image_fns.extend(list(data_path.glob('*.png'))) # Also include png files
image_fns = image_fns[:10]
print(f"Found {len(image_fns)} images to process from {data_path}")

for i, image_fn in enumerate(image_fns):
    print(f"Processing {i+1}/{len(image_fns)}: {image_fn.name}")
    
    try:
        image = Image.open(image_fn).convert("RGB") # Ensure image is RGB
        
        # Get model outputs, including attentions
        outputs = get_model_outputs(image, output_attentions=True)
        
        # Generate and save the distance visualization (now with foreground masking)
        distance_save_path = OUTPUT_DIR / f'dino-distance-vis_{i}.png'
        visualize_distance_map(image, outputs, distance_save_path)
        
        # Generate and save the attention visualization
        attention_save_path = OUTPUT_DIR / f'dino-attention-vis_{i}.png'
        visualize_attention_map(image, outputs, attention_save_path)
        
        # Generate and save the comparison visualization
        comparison_save_path = OUTPUT_DIR / f'dino-comparison-vis_{i}.png'
        visualize_comparison(image, outputs, comparison_save_path)
        
        # Log artifacts to MLflow if enabled
        if ENABLE_MLFLOW:
            mlflow.log_artifact(distance_save_path)
            mlflow.log_artifact(attention_save_path)
            mlflow.log_artifact(comparison_save_path)
            
    except Exception as e:
        print(f"Could not process {image_fn.name}. Error: {e}")

mask_save_path = OUTPUT_DIR / f'dino-mask-vis_{i}.png'
visualize_mask(image, outputs, mask_save_path)
if ENABLE_MLFLOW:
    mlflow.end_run()