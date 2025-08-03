#!/usr/bin/env python3

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np

# Connect to Milvus
print("Connecting to Milvus...")
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

print("Connected to Milvus successfully!")

# List all collections
print("\nExisting collections:")
collections = utility.list_collections()
for coll in collections:
    print(f"  - {coll}")

# Check if our collection exists
collection_name = "cls_tokens_collection"
if utility.has_collection(collection_name):
    print(f"\n✓ Collection '{collection_name}' exists!")
    
    # Get collection info
    collection = Collection(collection_name)
    print(f"Collection schema: {collection.schema}")
    print(f"Number of entities: {collection.num_entities}")
    
    # List partitions
    partitions = collection.partitions
    print(f"Partitions: {[p.name for p in partitions]}")
    
else:
    print(f"\n✗ Collection '{collection_name}' does not exist!")
    
    # Create the collection
    print(f"Creating collection '{collection_name}'...")
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="image_size", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="cls_token_dim", dtype=DataType.INT64),
        FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]
    
    schema = CollectionSchema(fields=fields, description="CLS tokens collection")
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    print(f"✓ Collection '{collection_name}' created successfully!")

# Test inserting a sample vector
print("\nTesting insertion...")
if utility.has_collection(collection_name):
    collection = Collection(collection_name)
    collection.load()
    
    # Sample data
    sample_embedding = np.random.rand(768).tolist()
    sample_data = [
        [sample_embedding],  # embedding
        ["test_image.jpg"],  # image_name
        ["/path/to/test_image.jpg"],  # image_path
        ["(100, 100)"],  # image_size
        [768],  # cls_token_dim
        ["clip-vit-base-patch16"],  # model
        ["Test image description"]  # text
    ]
    
    # Insert data
    collection.insert(sample_data)
    collection.flush()
    
    print(f"✓ Sample data inserted! Total entities: {collection.num_entities}")

print("\nDone!") 