from langchain.embeddings import HuggingFaceEmbeddings
import os
import uuid
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"


# Try importing ChromaDB safely
try:
    import chromadb
    chromadb_available = True
except ImportError:
    print("⚠️ ChromaDB import failed. Disabling embeddings.")
    chromadb_available = False

CHROMA_PATH = "./chroma_db"

def embed_store(nodes):
    """Stores embeddings in ChromaDB if available."""
    if not chromadb_available:
        print("⚠️ ChromaDB is unavailable. Skipping embedding storage.")
        return None

    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    documents = [getattr(node, "text", getattr(node, "page_content", node)) for node in nodes if isinstance(node, (str, object))]

    if not documents:
        raise ValueError("No valid text data found in nodes!")

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(name="chroma_collection")

        unique_ids = [str(uuid.uuid4()) for _ in documents]
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

        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results.get("documents", [])[0] if results.get("documents") else []
    except Exception as e:
        print(f"❌ Error querying ChromaDB: {e}")
        return []
