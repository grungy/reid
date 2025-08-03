import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPVisionModel, AutoImageProcessor
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.simple import SimpleVectorStore
import pickle
from typing import List, Dict, Any

class SimpleCLSTokenStore:
    def __init__(self, storage_path: str = "./cls_token_db"):
        """
        Initialize a simple CLS token storage system using LlamaIndex with local storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Load the CLIP model and processor
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Initialize simple vector store
        self.vector_store = SimpleVectorStore()
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Initialize index
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )
        
    def extract_cls_token(self, image: Image.Image) -> np.ndarray:
        """
        Extract CLS token from an image using CLIP
        """
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get CLS token (first token in the sequence) and convert to numpy
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token.squeeze().numpy()
    
    def process_image(self, image_path: Path, metadata: Dict[str, Any] = None) -> Document:
        """
        Process a single image and create a Document for storage
        """
        # Load image
        image = Image.open(image_path)
        
        # Extract CLS token
        cls_embedding = self.extract_cls_token(image)
        
        # Create document text
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
        try:
            document = self.process_image(image_path, metadata)
            self.index.insert(document)
            print(f"✓ Added image: {image_path.name}")
        except Exception as e:
            print(f"✗ Error adding {image_path.name}: {e}")
    
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
        for doc in documents:
            self.index.insert(doc)
        print(f"✓ Successfully added {len(documents)} images to database")
    
    def save_database(self, filename: str = "cls_token_database.pkl"):
        """
        Save the database to disk
        """
        save_path = self.storage_path / filename
        
        # Save the vector store data
        with open(save_path, 'wb') as f:
            pickle.dump(self.vector_store, f)
        
        print(f"✓ Database saved to: {save_path}")
    
    def load_database(self, filename: str = "cls_token_database.pkl"):
        """
        Load the database from disk
        """
        load_path = self.storage_path / filename
        
        if load_path.exists():
            with open(load_path, 'rb') as f:
                self.vector_store = pickle.load(f)
            
            # Recreate storage context and index
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
            print(f"✓ Database loaded from: {load_path}")
        else:
            print(f"✗ Database file not found: {load_path}")
    
    def get_all_images(self):
        """
        Retrieve all stored images
        """
        retriever = self.index.as_retriever(similarity_top_k=1000)
        nodes = retriever.retrieve("")
        return nodes
    
    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search for similar images using a CLS token embedding
        """
        # Create a query document with the embedding
        query_doc = Document(
            text="Query image",
            embedding=query_embedding
        )
        
        # Insert temporarily and search
        self.index.insert(query_doc)
        
        # Get similar documents
        retriever = self.index.as_retriever(similarity_top_k=top_k + 1)  # +1 to account for query doc
        nodes = retriever.retrieve("")
        
        # Remove the query document from results
        results = [node for node in nodes if node.text != "Query image"]
        
        return results[:top_k]

def main():
    # Initialize the storage system
    store = SimpleCLSTokenStore(storage_path="./cls_token_db")
    
    # Get image paths
    data_path = Path('./yellow_car/').expanduser()
    image_fns = list(data_path.glob('*.jpg'))
    image_fns = image_fns[:10]  # Limit for testing
    
    print(f"Found {len(image_fns)} images to process")
    print("="*50)
    
    # Add images to database
    for image_path in image_fns:
        metadata = {
            "category": "yellow_car",
            "dataset": "reid_dataset"
        }
        store.add_image(image_path, metadata)
    
    # Save the database
    store.save_database()
    
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
    
    # Example: Search for similar images
    if len(image_fns) > 1:
        print("="*50)
        print("SIMILARITY SEARCH EXAMPLE")
        print("="*50)
        
        # Use the first image as query
        query_image_path = image_fns[0]
        query_embedding = store.extract_cls_token(Image.open(query_image_path))
        
        print(f"Searching for images similar to: {query_image_path.name}")
        similar_images = store.search_by_embedding(query_embedding, top_k=3)
        
        print(f"Found {len(similar_images)} similar images:")
        for i, node in enumerate(similar_images):
            print(f"{i+1}. {node.metadata.get('image_name', 'Unknown')}")
            print(f"   Similarity score: {node.score:.4f}")

if __name__ == "__main__":
    main() 