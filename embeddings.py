from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import os
import uuid  # ✅ Generate unique IDs for each document

CHROMA_PATH = "./chroma_db"

def embed_store(nodes):
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    documents = []
    metadata_list = []

    for node in nodes:
        if hasattr(node, "text"):
            documents.append(node.text)
        elif hasattr(node, "page_content"):
            documents.append(node.page_content)
        elif isinstance(node, str):
            documents.append(node)
        else:
            print(f"⚠️ Warning: Unrecognized node type {type(node)}, skipping.")

    if not documents:
        raise ValueError("No valid text data found in nodes!")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="chroma_collection")

    embeddings = hf_embeddings.embed_documents(documents)

    # ✅ Ensure unique IDs to avoid overwriting old data
    unique_ids = [str(uuid.uuid4()) for _ in documents]

    collection.add(
        ids=unique_ids,
        documents=documents,
        embeddings=embeddings
    )

    return collection

def load_index():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(name="chroma_collection")

def query_index(collection, query, top_k=5):
    if not collection:
        return []
    
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    query_embedding = hf_embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0] if results.get("documents") else []
