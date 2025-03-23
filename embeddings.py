from langchain.embeddings import HuggingFaceEmbeddings
import os
import uuid  # ✅ Generate unique IDs for each document
import os

# Force ChromaDB to use system SQLite
os.environ["LD_LIBRARY_PATH"] = "/usr/lib"

try:
    import chromadb
    chromadb_available = True
except ImportError:
    print("⚠️ ChromaDB import failed. Disabling embeddings.")
    chromadb_available = False

# Try importing ChromaDB, handle failures
try:
    import chromadb
    chromadb_available = True
except ImportError:
    chromadb_available = False

CHROMA_PATH = "./chroma_db"

def embed_store(nodes):
    """Stores embeddings in ChromaDB if available."""
    if not chromadb_available:
        print("⚠️ ChromaDB is unavailable. Skipping embedding storage.")
        return None

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

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="chroma_collection")

    # Generate unique IDs to avoid overwriting old data
    unique_ids = [str(uuid.uuid4()) for _ in documents]

    try:
        embeddings = hf_embeddings.embed_documents(documents)
        collection.add(ids=unique_ids, documents=documents, embeddings=embeddings)
        return collection
    except Exception as e:
        print(f"❌ Error storing embeddings: {e}")
        return None

def load_index():
    """Loads ChromaDB index if available."""
    if not chromadb_available:
        print("⚠️ ChromaDB is unavailable. Cannot load index.")
        return None

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        return chroma_client.get_or_create_collection(name="chroma_collection")
    except Exception as e:
        print(f"❌ Error loading ChromaDB index: {e}")
        return None

def query_index(collection, query, top_k=5):
    """Queries the ChromaDB index safely."""
    if not chromadb_available or not collection:
        print("⚠️ ChromaDB is unavailable. Skipping query.")
        return []

    try:
        hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        query_embedding = hf_embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results["documents"][0] if results.get("documents") else []
    except Exception as e:
        print(f"❌ Error querying ChromaDB: {e}")
        return []
