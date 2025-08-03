import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import Dinov2Model, AutoImageProcessor
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_opening, binary_closing
import mlflow
import os
import requests

# --- Configuration ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

ENABLE_MLFLOW = os.getenv('ENABLE_MLFLOW', 'true').lower() == 'true'
OUTPUT_DIR = Path(__file__).parent / 'outputs_dino_sam_box' # New output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# --- MLflow Setup ---
if ENABLE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("dino-sam-box-prompt")
    mlflow.start_run()

# --- Model Loading ---
print("Loading DINOv2 model...")
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base", attn_implementation="eager")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
print("DINOv2 model loaded.")

# --- SAM Setup ---
def setup_sam_model():
    sam_checkpoint_path = Path("sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if not sam_checkpoint_path.exists():
        print("Downloading SAM checkpoint...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(sam_checkpoint_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")

    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM model loaded.")
    return predictor

sam_predictor = setup_sam_model()

# --- Helper Functions ---
def get_dino_outputs(image):
    inputs = dino_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = dino_model(**inputs, output_attentions=True)
    return outputs

def create_sam_point_prompt_from_dino_centroid(dino_outputs):
    """
    Calculates the center of mass of the attention map to create a robust
    point prompt for SAM.
    """
    attention_weights = dino_outputs.attentions[-1].mean(dim=1)
    cls_to_patch_attention = attention_weights[0, 0, 1:]
    
    grid_size = int(np.sqrt(cls_to_patch_attention.shape[0]))
    attention_map_2d = cls_to_patch_attention.reshape(grid_size, grid_size)

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    # Calculate the total attention sum
    total_attention = attention_map_2d.sum()

    if total_attention == 0:
        return None, None # Avoid division by zero

    # Calculate the weighted average of the coordinates
    center_y = (y_coords.float() * attention_map_2d).sum() / total_attention
    center_x = (x_coords.float() * attention_map_2d).sum() / total_attention

    # Round to get the nearest patch index
    prompt_row = int(torch.round(center_y))
    prompt_col = int(torch.round(center_x))

    # Convert grid coordinate to pixel coordinate
    stride = dino_processor.crop_size["height"] // grid_size
    prompt_coord_x = prompt_col * stride + (stride // 2)
    prompt_coord_y = prompt_row * stride + (stride // 2)

    input_point = np.array([[prompt_coord_x, prompt_coord_y]])
    input_label = np.array([1]) # 1 indicates a foreground point

    return input_point, input_label

def create_sam_box_prompt_from_dino(dino_outputs):
    """Creates a cleaned bounding box prompt from the DINOv2 attention map."""
    attention_weights = dino_outputs.attentions[-1].mean(dim=1)
    cls_to_patch_attention = attention_weights[0, 0, 1:]
    
    # 1. Create the initial noisy binary mask
    attention_threshold = torch.quantile(cls_to_patch_attention, 0.80)
    foreground_mask_1d = cls_to_patch_attention > attention_threshold
    
    grid_size = int(np.sqrt(cls_to_patch_attention.shape[0]))
    mask_2d = foreground_mask_1d.reshape(grid_size, grid_size).numpy()
    
    # --- Start of The Fix ---
    
    # 2. Apply opening to remove small background noise
    cleaned_mask_2d = binary_opening(mask_2d)
    
    # 3. (Optional) Apply closing to fill any holes in the main object
    cleaned_mask_2d = binary_closing(cleaned_mask_2d)
    
    # --- End of The Fix ---

    # 4. Find the coordinates of the cleaned foreground patches
    foreground_indices = torch.nonzero(torch.from_numpy(cleaned_mask_2d))
    if foreground_indices.numel() == 0:
        return None

    # Find the min/max row and column indices to form a box
    min_row, max_row = foreground_indices[:, 0].min(), foreground_indices[:, 0].max()
    min_col, max_col = foreground_indices[:, 1].min(), foreground_indices[:, 1].max()
    
    # Convert grid coordinates to pixel coordinates
    stride = dino_processor.crop_size["height"] // grid_size
    box_x_min = min_col * stride
    box_y_min = min_row * stride
    box_x_max = (max_col + 1) * stride
    box_y_max = (max_row + 1) * stride
    
    return np.array([box_x_min, box_y_min, box_x_max, box_y_max])

def visualize_sam_mask(image, sam_mask, point_prompt, save_path):
    """Visualizes the SAM mask and the point used as a prompt."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    show_mask(sam_mask, ax)
    
    # Draw the point prompt
    if point_prompt is not None:
        point_x, point_y = point_prompt[0]  # Extract coordinates from the point array
        ax.scatter(point_x, point_y, c='red', s=100, marker='o', edgecolors='white', linewidth=2)

    ax.axis('off')
    ax.set_title("SAM-Generated Mask with DINOv2 Point Prompt")
    plt.savefig(save_path)
    plt.close(fig)

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# --- Main Processing Loop ---
data_path = Path(__file__).parent.parent / 'yellow_car'
image_fns = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
image_fns = image_fns[:10]
print(f"Found {len(image_fns)} images to process from {data_path}")

for i, image_fn in enumerate(image_fns):
    print(f"Processing {i+1}/{len(image_fns)}: {image_fn.name}")
    
    try:
        image = Image.open(image_fn).convert("RGB")
        
        # 1. Get DINOv2 outputs
        dino_outputs = get_dino_outputs(image)
        
        # 2. Create a point prompt from DINOv2 centroid
        input_box = create_sam_box_prompt_from_dino(dino_outputs)
        input_point, input_label = create_sam_point_prompt_from_dino_centroid(dino_outputs)
        if input_point is None:
            print(f"Skipping {image_fn.name} due to no foreground patches found.")
            continue
            
        # 3. Run SAM with the point prompt
        sam_predictor.set_image(np.array(image))
        
        masks, scores, _ = sam_predictor.predict(
            box=input_box,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        sam_mask = masks[0]
        
        # 4. Visualize the result
        sam_mask_save_path = OUTPUT_DIR / f'sam-point-mask-vis_{i}.png'
        visualize_sam_mask(image, sam_mask, input_point, sam_mask_save_path)
        
        # 5. TODO: Use 'sam_mask' to filter patch embeddings
        
        if ENABLE_MLFLOW:
            mlflow.log_artifact(sam_mask_save_path)
            
    except Exception as e:
        print(f"Could not process {image_fn.name}. Error: {e}")

if ENABLE_MLFLOW and mlflow.active_run():
    mlflow.end_run()