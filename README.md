# Gen-Project
📄 AI-Powered Document Assistant with RAG, LangChain & Groq
🔹 An AI-driven document assistant that enables users to upload files, retrieve relevant content, and interact with a chatbot using Llama 3 via Groq.

🚀 Project Overview
This project is a Retrieval-Augmented Generation (RAG) based AI assistant that allows users to:
✅ Upload multiple documents
✅ Extract and store relevant information in a vector database
✅ Query the documents using natural language
✅ Get responses from Llama 3 (70B) hosted on Groq
✅ Enhance search with embeddings from Hugging Face

🔹 Features
✔️ Document Upload & Parsing: Uses Llama File Parser to extract structured content from uploaded files.
✔️ ChromaDB & SQLite3: Stores document embeddings and enables fast search.
✔️ Hugging Face Embeddings: Converts document text into numerical vectors for efficient similarity search.
✔️ Llama 3 (Groq): A freely accessible LLM to generate responses.
✔️ LangChain Integration: Manages context, retrieval, and response generation.
✔️ Streamlit UI: Simple and interactive user interface for seamless interaction.

🛠️ Tech Stack
LLM & AI: Groq (Llama 3), Transformers, LangChain

Embeddings: Hugging Face (BAAI/bge-small-en-v1.5)

Vector Database: ChromaDB, SQLite3

Backend: FastAPI

Frontend: Streamlit

📌 Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/your-repo.git
cd your-repo
2️⃣ Create a Virtual Environment & Install Dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3️⃣ Set Up Environment Variables
Create a .env file and add your Groq API Key:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key
4️⃣ Run the Application
bash
Copy
Edit
streamlit run main.py
🖼️ UI Preview
📌 [Include a screenshot of your app UI here]

🔗 Links
GitHub Repo: [Your Repo Link]

Live Demo: [Your Deployed App (if available)]

📜 License
This project is open-source under the MIT License.

📢 Contributions are welcome! Feel free to open an issue or submit a PR. 🚀

