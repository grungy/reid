import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPVisionModel, AutoImageProcessor
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage import StorageContext
import qdrant_client
import os
from typing import List, Dict, Any

class CLSTokenStore:
    def __init__(self, collection_name: str = "image_cls_tokens"):
        """
        Initialize the CLS token storage system using LlamaIndex and Qdrant
        """
        self.collection_name = collection_name
        
        # Load the CLIP model and processor
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Initialize Qdrant client (in-memory for this example)
        self.client = qdrant_client.QdrantClient(":memory:")
        
        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Initialize index
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )
        
        # Store for batch processing
        self.documents = []
        
    def extract_cls_token(self, image: Image.Image) -> torch.Tensor:
        """
        Extract CLS token from an image using CLIP
        """
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get CLS token (first token in the sequence)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token
    
    def process_image(self, image_path: Path, metadata: Dict[str, Any] = None) -> Document:
        """
        Process a single image and create a Document for storage
        """
        # Load image
        image = Image.open(image_path)
        
        # Extract CLS token
        cls_token = self.extract_cls_token(image)
        
        # Convert to numpy array for storage
        cls_embedding = cls_token.squeeze().numpy()
        
        # Create document text (you can customize this)
        doc_text = f"Image: {image_path.name}\nSize: {image.size}\nCLS Token Dimension: {cls_embedding.shape[0]}"
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_size": image.size,
            "cls_token_dim": cls_embedding.shape[0],
            "model": "clip-vit-base-patch16"
        })
        
        # Create LlamaIndex Document
        document = Document(
            text=doc_text,
            metadata=metadata,
            embedding=cls_embedding
        )
        
        return document
    
    def add_image(self, image_path: Path, metadata: Dict[str, Any] = None):
        """
        Add a single image to the database
        """
        document = self.process_image(image_path, metadata)
        self.index.insert(document)
        print(f"Added image: {image_path.name}")
    
    def add_images_batch(self, image_paths: List[Path], metadata_list: List[Dict[str, Any]] = None):
        """
        Add multiple images to the database in batch
        """
        if metadata_list is None:
            metadata_list = [None] * len(image_paths)
        
        documents = []
        for i, (image_path, metadata) in enumerate(zip(image_paths, metadata_list)):
            try:
                document = self.process_image(image_path, metadata)
                documents.append(document)
                print(f"Processed {i+1}/{len(image_paths)}: {image_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
        
        # Insert all documents at once
        self.index.insert_nodes(documents)
        print(f"Successfully added {len(documents)} images to database")
    
    def search_similar_images(self, query_image_path: Path, top_k: int = 5):
        """
        Search for similar images using CLS token similarity
        """
        # Process query image
        query_document = self.process_image(query_image_path)
        
        # Create a temporary index for the query
        query_index = VectorStoreIndex.from_documents(
            [query_document],
            storage_context=self.storage_context
        )
        
        # Perform similarity search
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query("Find similar images")
        
        return response
    
    def get_all_images(self):
        """
        Retrieve all stored images
        """
        retriever = self.index.as_retriever(similarity_top_k=1000)  # Large number to get all
        nodes = retriever.retrieve("")
        return nodes

def main():
    # Initialize the storage system
    store = CLSTokenStore(collection_name="yellow_car_cls_tokens")
    
    # Get image paths
    data_path = Path('./yellow_car/').expanduser()
    image_fns = list(data_path.glob('*.jpg'))  # Adjust pattern as needed
    image_fns = image_fns[:10]  # Limit for testing
    
    print(f"Found {len(image_fns)} images to process")
    
    # Add images to database
    for image_path in image_fns:
        metadata = {
            "category": "yellow_car",
            "dataset": "reid_dataset"
        }
        store.add_image(image_path, metadata)
    
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)
    
    # Get all stored images
    all_nodes = store.get_all_images()
    print(f"Total images in database: {len(all_nodes)}")
    
    # Show some examples
    print("\nSample stored images:")
    for i, node in enumerate(all_nodes[:3]):
        print(f"{i+1}. {node.metadata.get('image_name', 'Unknown')}")
        print(f"   Size: {node.metadata.get('image_size', 'Unknown')}")
        print(f"   CLS Token Dim: {node.metadata.get('cls_token_dim', 'Unknown')}")
        print()
    
    # Example similarity search
    if len(image_fns) > 1:
        print("="*50)
        print("SIMILARITY SEARCH EXAMPLE")
        print("="*50)
        
        query_image = image_fns[0]
        print(f"Searching for images similar to: {query_image.name}")
        
        # Note: This is a simplified example. In practice, you'd want to use
        # the embedding directly for similarity search
        similar_nodes = store.get_all_images()
        print(f"Found {len(similar_nodes)} total images in database")

if __name__ == "__main__":
    main() 