# Multi-Agent LangGraph System: End-to-End Implementation

## Folder Structure
```
MultiAgentLangGraph/
├── data/
│   ├── database/
│   │   └── claims.db                  # SQLite DB with multiple domain tables
│   ├── documents/
│   │   └── *.pdf                      # PDF documents per line of business
│   ├── images/
│   │   └── *.png                      # Flowchart-like data diagrams per domain
├── vectorstores/
│   ├── pdf_faiss/                     # FAISS vector store for PDFs
│   └── image_faiss/                   # FAISS vector store for OCR'd images
├── langgraph_app/
│   ├── agents/
│   │   ├── router_agent.py
│   │   ├── sql_agent.py
│   │   ├── pdf_rag_agent.py
│   │   ├── ocr_rag_agent.py
│   │   ├── human_critic_agent.py
│   │   └── summarizer_agent.py
│   ├── graph_flow.py
│   └── app.py                         # Entry point
├── utils/
│   ├── create_dummy_data.py          # Generates dummy DB, PDFs, and images
│   ├── pdf_vector_builder.py         # Vector DB for PDFs
│   ├── image_vector_builder.py       # OCR + vector DB for images
├── Dockerfile
├── docker-compose.yml
├── README.md
```

---

## ✅ DONE: Step 1 - Dummy Data Creation
- `utils/create_dummy_data.py`
  - Creates SQLite DB with 5 tables (Property, Casualty, Banking, Automobile, Technical)
  - Generates 5 PDF files with sample paragraphs
  - Generates 5 flowchart-style PNG diagrams with annotated boxes

---

## 🔄 Step 2 - Vector DB for PDFs
**`utils/pdf_vector_builder.py`**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

pdf_dir = "data/documents"
vector_store_path = "vectorstores/pdf_faiss"
os.makedirs(vector_store_path, exist_ok=True)

docs = []
for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, file))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local(vector_store_path)
```

---

## 🔄 Step 3 - OCR & Vector DB for Images
**`utils/image_vector_builder.py`**
```python
from PIL import Image
import pytesseract
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

image_dir = "data/images"
vector_store_path = "vectorstores/image_faiss"
os.makedirs(vector_store_path, exist_ok=True)

documents = []
for img_file in os.listdir(image_dir):
    if img_file.endswith(".png"):
        img_path = os.path.join(image_dir, img_file)
        text = pytesseract.image_to_string(Image.open(img_path))
        documents.append(Document(page_content=text, metadata={"source": img_file}))

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(vector_store_path)
```

---

## 🔄 Step 4 - Agents & LangGraph Workflow
**`langgraph_app/agents/`**
Each agent will:
- Take `input_query`
- Respond with its `response_text`
- Pass to the next agent or return

**Agents Include:**
- `router_agent.py`: Classifies query (SQL / PDF / Image)
- `sql_agent.py`: Runs SQL query and creates paragraph output
- `pdf_rag_agent.py`: Searches FAISS PDF store for answer
- `ocr_rag_agent.py`: Searches FAISS Image OCR store
- `human_critic_agent.py`: Reviews the output
- `summarizer_agent.py`: Final output polishing

**Graph DAG (`langgraph_app/graph_flow.py`)**
- Implements LangGraph DAG using `StateGraph`
- Nodes: Router → (SQL or PDF or OCR) → Critic → Summarizer → Final
- State carries: `query`, `response`, `source_type`, `metadata`

---

## ✅ Step 5 - Dockerization

### 🔧 `Dockerfile`
```Dockerfile
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (for FastAPI/Streamlit later)
EXPOSE 8501

# Run entry script
CMD ["python", "langgraph_app/app.py"]
```

### 🛠️ `docker-compose.yml`
```yaml
version: '3.9'
services:
  langgraph-agent:
    build: .
    container_name: langgraph_container
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./vectorstores:/app/vectorstores
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

### 📄 `.dockerignore`
```
__pycache__
*.pyc
*.pyo
*.pyd
*.db
*.sqlite3
*.log
venv
.env
```

---

## 🔄 Step 6 - Streamlit / FastAPI UI
Will allow real-time user queries and trace display.

---

✅ Docker setup complete!
