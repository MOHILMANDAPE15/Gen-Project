# Gen-Project

# Self-Made GPT

Welcome to Self-Made GPT! This is a chatbot project I built using Streamlit and Groq's LLaMA 3 model. It allows users to upload files, generate embeddings, and query them to make responses more intelligent and context-aware.

## What This Project Does
- Uses the `llama3-70b-8192` model for smart conversations.
- Lets users upload files (PDF, TXT, DOCX) to add context to the chat.
- Processes and extracts meaningful information using `LlamaParse`.
- Stores and retrieves embeddings using ChromaDB for efficient searching.
- Provides an interactive chat interface with a message history.

## How to Install
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/self-made-gpt.git
   cd self-made-gpt
   ```
2. Install all the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up API keys in a `.env` file:
   ```
   GROQ_API_KEY=your_api_key_here
   LLAMA_CLOUD_API_KEY=your_llama_api_key_here
   ```

## How to Use It
Run the application with:
```sh
streamlit run main.py
```
Once it's running, you can upload files, ask questions, and the chatbot will respond using the uploaded documents as a knowledge base.

## How File Processing Works
- When a file is uploaded, it's parsed using `LlamaParse` to extract readable text.
- The parsed text is then chunked and analyzed to pull out key titles and potential questions.
- Embeddings are created using `HuggingFaceEmbeddings` and stored in ChromaDB.
- When a user asks a question, the chatbot searches for relevant content in the stored embeddings and uses that context to improve responses.

## My Code Structure
- `main.py`: Runs the Streamlit app and manages chat history.
- `file_uploader.py`: Handles file uploads and ensures files are stored properly.
- `file_parser.py`: Parses files, extracts content, and processes them for embeddings.
- `embeddings.py`: Generates embeddings, stores them in ChromaDB, and handles queries efficiently.

## Dependencies Used
- `streamlit`
- `streamlit_chat`
- `langchain`
- `chromadb`
- `llama_parse`
- `python-dotenv`

## License
This project is licensed under the MIT License.

---

I built this project to experiment with embedding-based search and improve chatbot responses with contextual data. Feel free to fork it, improve it, or suggest any changes!

