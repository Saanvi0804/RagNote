RagNote — Local PDF Question Answering (Ollama + FastAPI)

RagNote is a private, local RAG (Retrieval-Augmented Generation) tool that lets you upload a PDF and ask questions about it.
It runs completely offline on your device, using:

FastAPI (backend)

FAISS for embeddings

Ollama + a local model like phi3 for generation

React (frontend UI)

No cloud API keys, no billing — your PDF never leaves your machine.

🚀 Features

📄 Upload any PDF

🔍 Extracts and embeds document text

🤖 Answers questions using hybrid RAG + model knowledge

🛡 Fully local & privacy-safe

⚡ Fast response using FAISS + local LLM

🎨 Clean React UI

🔧 Tech Stack
Layer	Tech
Backend	FastAPI, Python, FAISS, SentenceTransformers
Model	Ollama (e.g., phi3, mistral)
Frontend	React + Axios
Local AI	No OpenAI or cloud APIs required
📦 Setup Instructions
1️⃣ Install Dependencies
Backend
cd backend/app
pip install -r requirements.txt

Frontend
cd frontend
npm install

2️⃣ Run Ollama

Download a model (example: phi3):

ollama pull phi3


Start Ollama (usually auto-starts):

ollama list

3️⃣ Start the Backend
cd backend/app
uvicorn main:app --reload --host 127.0.0.1 --port 8000

4️⃣ Start the Frontend
cd frontend
npm start


Frontend → http://localhost:3000

Backend → http://localhost:8000

🖼 How It Works

You upload a PDF

RagNote extracts text

Builds embeddings using MiniLM

Stores them in a FAISS index

When you ask a question:

Retrieves top matching chunks

Passes them + your question to Ollama

Model generates a clean answer
