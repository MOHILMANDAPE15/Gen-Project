import nest_asyncio
nest_asyncio.apply()
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

def parsing():
    load_dotenv()
    parse_key = os.getenv("LLAMA_CLOUD_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not parse_key:
        st.error("Please set the environment variable LLAMA_CLOUD_API_KEY in .env")
        st.stop()

    # Get uploaded files
    file_paths = st.session_state.get("uploaded_files", [])
    if not file_paths:
        st.warning("No files uploaded.")
        return st.session_state.get("processed_nodes", [])  # Return stored nodes if no new files

    # Persist processed files across reruns
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "processed_nodes" not in st.session_state:
        st.session_state.processed_nodes = []

    # Identify new files (avoid reprocessing old ones)
    new_files = [file for file in file_paths if file not in st.session_state.processed_files]
    if not new_files:
        st.info("All uploaded files have already been processed.")
        return st.session_state.processed_nodes

    # Process new files
    parser = LlamaParse(result_type="text", api_key=parse_key)
    processed_docs = []

    for file_path in new_files:
        st.write(f"Processing new file: {file_path}")
        docs = parser.load_data(file_path)
        processed_docs.extend(docs)
        os.remove(file_path)  # Remove after processing
        st.session_state.processed_files.add(file_path)  # Persist processed files

    st.session_state.uploaded_files = []  # Clear uploaded files after processing
    st.write("Processing complete!")

    # Extract text and metadata
    llm = Groq(model='llama3-70b-8192', api_key=groq_key)
    text_splitter = SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
    title_extractor = TitleExtractor(llm=llm, nodes=5)
    qa_extractor = QuestionsAnsweredExtractor(llm=llm, questions=3)

    pipeline = IngestionPipeline(transformations=[text_splitter, title_extractor, qa_extractor])
    nodes = pipeline.run(documents=processed_docs, in_place=True, show_progress=True)

    # Append new nodes to existing ones
    st.session_state.processed_nodes.extend(nodes)
    
    return st.session_state.processed_nodes
