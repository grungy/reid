import torch
import torch.nn.functional as F

# Assuming you have your tensors from the previous step
# cls_token shape: (1, 768)
# patch_tokens shape: (1, 196, 768)
# (Using example shapes for ViT-B/16)

# Calculate Cosine Similarity between the CLS token and each patch token
# We need to add a dimension to cls_token to allow broadcasting
cosine_similarities = F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)

# Convert similarity to distance. Distance = 1 - Similarity
cosine_distances = 1 - cosine_similarities

# The result is a tensor of distances, one for each patch
print(f"Shape of distances tensor: {cosine_distances.shape}")
# Example output: Shape of distances tensor: torch.Size([1, 196])

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import glob

# --- Assume you have these variables from previous steps ---
# cosine_distances: The tensor of distances. Shape: (1, 196)
# image: The original PIL image.
# patch_tokens: The patch embeddings tensor. Shape: (1, 196, 768)

# image = Image.open(requests.get(url, stream=True).raw).resize((224, 224))
data_path = '~/Data/VeRi/image_test/'
image_fns = glob.glob('./*.jpg')
print("image_fns: ", image_fns)
images = [Image.open(image_fn) for image_fn in image_fns]

for i, image in enumerate(images):
  image = Image.open('/content/0002_c002_00030600_0.jpg')
  print(image.size)
  # Dummy data for demonstration
  cls_token = torch.randn(1, 768)
  patch_tokens = torch.randn(1, 196, 768)
  cosine_similarities = F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)
  cosine_distances = 1 - cosine_similarities
  # --- End of example setup ---


  # 1. Determine the grid size of the patches
  num_patches = patch_tokens.shape[1]
  grid_size = int(np.sqrt(num_patches)) # e.g., sqrt(196) = 14

  # 2. Reshape the 1D distances into a 2D grid
  distance_map = cosine_distances.reshape(grid_size, grid_size)

  # 3. Resize the small distance map to the original image size for overlay
  # We convert to a tensor to use the efficient interpolate function
  distance_map_resized = torch.nn.functional.interpolate(
      distance_map.unsqueeze(0).unsqueeze(0),
      size=image.size,
      mode='bilinear',
      align_corners=False
  ).squeeze()

  # 4. Plot the visualization
  fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  ax.imshow(image) # Show the original image
  # Overlay the heatmap
  heatmap = ax.imshow(distance_map_resized.numpy(), cmap='viridis', alpha=0.5)
  fig.colorbar(heatmap, ax=ax)
  ax.axis('off')
  ax.set_title("Outlier Patch Visualization (Brighter = More Unique)")
  plt.show()
  plt.savefig('./vis_' + str(i))