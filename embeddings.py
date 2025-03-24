from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import os

CHROMA_PATH = "./chroma_db"

def embed_store(nodes):
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # ✅ Extract actual text content from nodes
    documents = []
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

    # ✅ Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="chroma_collection")

    # Generate embeddings
    embeddings = hf_embeddings.embed_documents(documents)

    # Add to ChromaDB
    collection.add(
        ids=[str(i) for i in range(len(documents))],  # Unique IDs
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

    return results["documents"][0] if "documents" in results and results["documents"] else []