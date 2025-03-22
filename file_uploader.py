import streamlit as st
import os

UPLOADS_DIR = "uploads"

def handle_file_upload():
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)

    # Ensure only one uploader instance is created
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True,
        key="file_uploader_main"  # ✅ Unique key
    )

    if uploaded_files:
        new_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)

            if file_path not in st.session_state.uploaded_files:  # Avoid duplicates
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files.append(file_path)

        st.session_state.uploaded_files.extend(new_files)  # Append new files
        return st.session_state.uploaded_files

    return st.session_state.uploaded_files
