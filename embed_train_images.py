import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPVisionModel, AutoImageProcessor
from matplotlib.colors import LogNorm
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.embeddings import BaseEmbedding
from typing import List, Optional
import pickle

# Custom embedding class that uses your CLS tokens
class CLSTokenEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        # For text queries, you might want to use a different approach
        # For now, we'll return zeros - you can modify this based on your needs
        return [0.0] * 768  # CLIP embedding dimension
    
    def _get_text_embedding(self, text: str) -> List[float]:
        # For text embeddings, you might want to use a different approach
        # For now, we'll return zeros - you can modify this based on your needs
        return [0.0] * 768  # CLIP embedding dimension
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # For batch text embeddings
        return [self._get_text_embedding(text) for text in texts]
    
    # Async versions of the methods (required by BaseEmbedding)
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

# Load the vision model and processor
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Initialize Milvus vector store for storing CLS tokens
# Connect to your Milvus instance
vector_store = MilvusVectorStore(
    collection_name="cls_tokens_collection",
    dim=768,  # CLIP ViT-Base has 768-dimensional embeddings
    host="localhost",
    port="19530"
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create custom embedding instance
custom_embedding = CLSTokenEmbedding()

# Initialize index with custom embedding
index = VectorStoreIndex.from_vector_store(
    vector_store, 
    storage_context=storage_context,
    embed_model=custom_embedding
)

# Get image paths
data_path = Path('./yellow_car/').expanduser()
image_fns = list(data_path.glob('*.jpg'))
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
documents = []
for i, image_fn in enumerate(image_fns):
    print(f"Processing {i+1}/{len(image_fns)}: {image_fn.name}")
    
    # Load and process image
    image = Image.open(image_fn)
    all_tokens, cls_token, patch_tokens = extract_tokens(image)
    cls_embedding = cls_token.squeeze().numpy()
    
    # Create document for storage
    doc_text = f"Image: {image_fn.name}\nSize: {image.size}\nCLS Token Dimension: {cls_embedding.shape[0]}"
    
    metadata = {
        "image_path": str(image_fn),
        "image_name": image_fn.name,
        "image_size": image.size,
        "cls_token_dim": cls_embedding.shape[0],
        "model": "clip-vit-base-patch16"
    }
    
    # Create LlamaIndex Document with your custom embedding
    document = Document(
        text=doc_text,
        metadata=metadata,
        embedding=cls_embedding.tolist()  # Convert to list for LlamaIndex
    )
    
    # Add to index
    index.insert(document)
    documents.append(document)
    
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
    plt.close()

print(f"\n✓ Successfully stored CLS tokens for {len(documents)} images in Milvus")

# Show summary of what was stored
print("\n" + "="*50)
print("STORAGE SUMMARY")
print("="*50)

print(f"Total images processed: {len(documents)}")
print(f"Collection name: cls_tokens_collection")
print(f"Embedding dimension: 768 (CLIP ViT-Base)")

# Show some examples of what was stored
print("\nSample stored images:")
for i, doc in enumerate(documents[:3]):
    print(f"{i+1}. {doc.metadata.get('image_name', 'Unknown')}")
    print(f"   Size: {doc.metadata.get('image_size', 'Unknown')}")
    print(f"   CLS Token Dim: {doc.metadata.get('cls_token_dim', 'Unknown')}")
    print(f"   Embedding length: {len(doc.embedding)}")
    print()

# Example: Direct similarity search using Milvus client
if len(image_fns) > 1:
    print("="*50)
    print("DIRECT MILVUS SIMILARITY SEARCH EXAMPLE")
    print("="*50)
    
    try:
        # Use the first image as query
        query_image_path = image_fns[0]
        query_image = Image.open(query_image_path)
        _, query_cls_token, _ = extract_tokens(query_image)
        query_embedding = query_cls_token.squeeze().numpy().tolist()
        
        print(f"Searching for images similar to: {query_image_path.name}")
        print(f"Query embedding dimension: {len(query_embedding)}")
        
        # Use the Milvus client directly for search
        search_results = vector_store.client.search(
            collection_name="cls_tokens_collection",
            data=[query_embedding],
            limit=5,
            output_fields=["image_name", "image_path"]
        )
        
        print(f"Found {len(search_results[0])} similar images:")
        for i, result in enumerate(search_results[0]):
            print(f"{i+1}. Score: {result.score:.4f}")
            print(f"   Image: {result.entity.get('image_name', 'Unknown')}")
            print()
            
    except Exception as e:
        print(f"Search example failed (this is normal for first run): {e}")
        print("The embeddings are still stored in Milvus and can be queried later.")

print("\n✓ All done! CLS tokens are now stored in Milvus database.")
print("Collection name: cls_tokens_collection")
print("Using custom CLS token embeddings without OpenAI dependency!")
print("\nYou can now query the collection directly using Milvus client or LlamaIndex.")