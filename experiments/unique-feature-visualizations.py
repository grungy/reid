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

# Set these environment variables before creating MLflow client
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
os.environ['ENABLE_MLFLOW'] = 'true'

# Toggle for running experiments
ENABLE_MLFLOW = os.getenv('ENABLE_MLFLOW', 'true').lower() == 'true'

if ENABLE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("feature-visualizations")
    mlflow.start_run()

# Load the vision model and processor
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Get image paths
data_path = Path(__file__).parent.parent / 'yellow_car'
print(data_path)
image_fns = list(data_path.glob('*.jpg'))
print(image_fns)
image_fns = image_fns[:10]  # Limit for testing
print(f"Found {len(image_fns)} images to process")

# Function to extract CLS token from an image
def extract_tokens(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Get CLS token (first token in the sequence)
    all_tokens = outputs.last_hidden_state
    cls_token = all_tokens[:, 0, :]
    patch_tokens = all_tokens[:, 1:, :]
    return all_tokens, cls_token, patch_tokens 

# Process each image and store CLS tokens
for i, image_fn in enumerate(image_fns):
    print(f"Processing {i+1}/{len(image_fns)}: {image_fn.name}")
    
    # Load and process image
    image = Image.open(image_fn)
    all_tokens, cls_token, patch_tokens = extract_tokens(image)
    
    cosine_similarities = F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)
    cosine_distances = 1 - cosine_similarities
    
    print(f"Shape of distances tensor: {cosine_distances.shape}")
    
    num_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(num_patches))
    print("num_patches: ", num_patches)
    print("grid_size: ", grid_size)
    
    distance_map = cosine_distances.reshape(grid_size, grid_size)
    viz_size = processor.crop_size["height"]
    resized_image = image.resize((viz_size, viz_size))
    grayscale_image = resized_image.convert('L')
    
    distance_map_resized = torch.nn.functional.interpolate(
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
    plt.savefig(f'./outputs/vis_{i}.png')
    if ENABLE_MLFLOW:
        mlflow.log_artifact(f'./outputs/vis_{i}.png')
    plt.close()

if ENABLE_MLFLOW:
    mlflow.end_run()