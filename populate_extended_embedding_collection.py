#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPVisionModel, AutoImageProcessor
from pymilvus import connections, Collection, utility

# Load the vision model and processor
print("Loading CLIP model...")
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Connect to Milvus
print("Connecting to Milvus...")
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# Get collection
collection_name = "extended_embedding_tokens_collection"
if not utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' does not exist!")
    exit(1)

collection = Collection(collection_name)
collection.load()

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

# Get image paths
data_path = Path('~/data/VeRi/image_train/').expanduser()
image_fns = list(data_path.glob('*.jpg'))
image_fns = image_fns[:100]  # Limit for testing
print(f"Found {len(image_fns)} images to process")

# Process each image and store CLS tokens
embeddings = []
image_names = []
image_paths = []
image_sizes = []
cls_token_dims = []
models = []
texts = []

for i, image_fn in enumerate(image_fns):
    print(f"Processing {i+1}/{len(image_fns)}: {image_fn.name}")
    
    # Load and process image
    image = Image.open(image_fn)
    all_tokens, cls_token, patch_tokens = extract_tokens(image)
    
    # Prepare data
    embeddings.append(cls_embedding.tolist())
    image_names.append(image_fn.name)
    image_paths.append(str(image_fn))
    image_sizes.append(str(image.size))
    cls_token_dims.append(cls_embedding.shape[0])
    models.append("clip-vit-base-patch16")
    texts.append(f"Image: {image_fn.name}\nSize: {image.size}\nCLS Token Dimension: {cls_embedding.shape[0]}")

# Insert data into collection
print(f"\nInserting {len(embeddings)} embeddings into collection...")
data = [
    embeddings,  # embedding
    image_names,  # image_name
    image_paths,  # image_path
    image_sizes,  # image_size
    cls_token_dims,  # cls_token_dim
    models,  # model
    texts  # text
]

collection.insert(data)
collection.flush()

print(f"✓ Successfully inserted {len(embeddings)} embeddings!")
print(f"Total entities in collection: {collection.num_entities}")

# Test search
print("\nTesting similarity search...")
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
results = collection.search(
    data=[embeddings[0]],  # Search using first embedding
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["image_name", "image_path"]
)

print(f"Search results for '{image_names[0]}':")
for i, result in enumerate(results[0]):
    print(f"{i+1}. Score: {result.score:.4f}")
    print(f"   Image: {result.entity.get('image_name', 'Unknown')}")

print("\nDone!") 