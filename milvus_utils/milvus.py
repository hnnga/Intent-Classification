from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
import numpy as np

COLLECTION_NAME = "questions_collection"
DIMENSION = 384  

def connect_milvus():
    try:
        connections.connect(host="localhost", port="19530")
        print("Milvus connected successfully")
    except Exception as e:
        print(f"Error: {e}")

def has_collection():
    return utility.has_collection(COLLECTION_NAME)

def create_collection():
    if has_collection():
        print(f"Collection {COLLECTION_NAME} is existed.")
        return

    fields = [
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=512)
    ]

    schema = CollectionSchema(fields, description="Collection for user questions")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})
    collection.load()
    # print(f"Collection {COLLECTION_NAME} has created.")

def insert_data(question_list, embeddings, intent):
    collection = Collection(name=COLLECTION_NAME)
    data = [
        question_list,
        embeddings,
        [intent] * len(question_list)
    ]
    collection.insert(data)
    collection.flush()
    print(f" Inserted {len(question_list)} to {COLLECTION_NAME}.")

def clear_collection():
    try:
        collection = Collection(name=COLLECTION_NAME)
        collection.release()  # Release before drop (safe cleanup)
        collection.drop()
        print(f"Collection '{COLLECTION_NAME}' has been deleted.")

        # Optionally, recreate the collection after dropping
        create_collection()
        print(f"Collection '{COLLECTION_NAME}' has been recreated and is now empty.")
    except Exception as e:
        print(f"Error while clearing collection: {e}")

# Test the connection and collection creation
connect_milvus()
# clear_collection()
create_collection()
