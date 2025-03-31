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
        return []

    processed_files = st.session_state.get("processed_files", set())
    new_files = [file for file in file_paths if file not in processed_files]

    if not new_files:
        st.info("All uploaded files have already been processed.")
        return st.session_state.get("processed_nodes", [])

    parser = LlamaParse(result_type="text", api_key=parse_key)
    processed_docs = []

    for file_path in new_files:
        st.write(f"Processing new file: {file_path}")
        docs = parser.load_data(file_path)
        processed_docs.extend(docs)
        os.remove(file_path) 
        processed_files.add(file_path) 

    st.session_state.uploaded_files = []
    st.session_state.processed_files = processed_files  
    st.write("Processing complete!")

 
    llm = Groq(model='llama3-70b-8192', api_key=groq_key)
    text_splitter = SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
    title_extractor = TitleExtractor(llm=llm, nodes=5)
    qa_extractor = QuestionsAnsweredExtractor(llm=llm, questions=3)

    pipeline = IngestionPipeline(transformations=[text_splitter, title_extractor, qa_extractor])
    nodes = pipeline.run(documents=processed_docs, in_place=True, show_progress=True)

    st.session_state.processed_nodes = st.session_state.get("processed_nodes", []) + nodes
    return st.session_state.processed_nodes
