import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from file_uploader import handle_file_upload
from file_parser import parsing
import asyncio
import sqlite3

# Set event loop (Fix for Streamlit async issues)
asyncio.set_event_loop(asyncio.new_event_loop())

# Handle ChromaDB Import (Avoid crashes if missing)
try:
    from embeddings import embed_store, load_index, query_index
    chromadb_available = True
except ImportError:
    st.warning("ChromaDB is unavailable. File search will be disabled.")
    chromadb_available = False


def init():
    """Initialize the app and check for API key."""
    st.set_page_config(page_title="Self-Made GPT", page_icon="🧊")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        api_key = st.text_input("Enter your GROQ API Key:", type="password")
        if not api_key:
            st.error("API key is required to proceed!")
            st.stop()


def main():
    init()
    st.title("Self-Made GPT")

    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.5)

    if "msg_history" not in st.session_state:
        st.session_state.msg_history = [SystemMessage(content="You are a helpful assistant.")]

    st.sidebar.header("File Upload")
    file_paths = handle_file_upload()

    if file_paths and chromadb_available:
        nodes = parsing()

        if nodes:
            st.write("Creating embeddings and storing in ChromaDB...")
            st.session_state.index = embed_store(nodes)  # ✅ Store index in session
            st.success("Embeddings created and stored!")

    # ChromaDB Index Handling
    if "index" not in st.session_state:
        if chromadb_available:
            sqlite_version = sqlite3.sqlite_version_info
            if sqlite_version < (3, 35, 0):
                st.warning("SQLite version is too old for ChromaDB. File search is disabled.")
                st.session_state.index = None  # Prevents crashes
            else:
                try:
                    st.session_state.index = load_index()  # ✅ Load index if it exists
                except Exception:
                    st.warning("Failed to load ChromaDB index. File-based search is disabled.")
                    st.session_state.index = None
        else:
            st.session_state.index = None

    user_input = st.text_input("Your message:", key="user_input")

    if user_input:
        st.session_state.msg_history.append(HumanMessage(content=user_input))

        # Handle query safely (Skip if ChromaDB is unavailable)
        context = ""
        if st.session_state.index:
            relevant_nodes = query_index(st.session_state.index, user_input, top_k=5)
            context = "\n".join(relevant_nodes)

        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {user_input}"),
        ])
        response_text = response.content if hasattr(response, "content") else str(response)

        st.session_state.msg_history.append(AIMessage(content=response_text))

    with st.sidebar:
        st.write("Chat History")
        for msg in st.session_state.msg_history:
            if isinstance(msg, HumanMessage):
                st.write(f"🧑‍💻 You: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.write(f"🤖 Bot: {msg.content}")

    for i, msg in enumerate(st.session_state.msg_history):
        if isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=f"user_{i}")
        elif isinstance(msg, AIMessage):
            message(msg.content, key=f"bot_{i}")


if __name__ == "__main__":
    main()
