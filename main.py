import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from file_uploader import handle_file_upload
from file_parser import parsing
from embeddings import embed_store, load_index, query_index

def init():
    st.set_page_config(page_title="DocMate", page_icon="🧊")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("Please set the environment variable GROQ_API_KEY in .env")
        st.stop()

def main():
    init()
    st.title("Self-Made GPT")
    
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.5)

    if "msg_history" not in st.session_state:
        st.session_state.msg_history = [SystemMessage(content="You are a helpful assistant.")]

    st.sidebar.header("File Upload")
    file_paths = handle_file_upload()

    if file_paths:
        nodes = parsing()

        if nodes:
            st.write("Creating embeddings and storing in chromadb..")
            st.session_state.index = embed_store(nodes)  # ✅ Store index in session
            st.success("Embeddings created and stored!")

    if "index" not in st.session_state:
        st.session_state.index = load_index()  

    user_input = st.text_input("Your message:", key="user_input")

    if user_input:
        st.session_state.msg_history.append(HumanMessage(content=user_input))

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
