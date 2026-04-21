# 💧 Hydration_RAG_Assistant

About: Built a Retrieval-Augmented Generation (RAG) assistant using FAISS and Sentence Transformers to answer hydration-related queries from PDF documents. Implemented semantic search, optimized chunking strategies, and integrated Google Gemini API for context-aware responses via a Streamlit interface.

## 🚀 Features
- Extracts and processes content from PDF documents
- Semantic search using Sentence Transformers
- Fast similarity search with FAISS
- Context-aware answers using Google Gemini API
- Interactive UI built with Streamlit
- Persistent FAISS index for faster reloads
- Restricts responses strictly to document context

## 🛠️ Tech Stack
- Python
- Streamlit
- FAISS
- Sentence Transformers
- Google Gemini API
- PyPDF

## 🧠 How It Works
1. **Document Loading**
  Reads and extracts text from the PDF file
2. **Text Chunking**
  Splits text into smaller overlapping chunks
3. **Embedding Generation**
  Converts chunks into vector embeddings
4. **FAISS Indexing**
  Stores embeddings for fast similarity search
5. **Query Processing**
  User query is embedded and matched against stored vectors
6. **Context Retrieval**
  Top relevant chunks are retrieved
7. **Answer Generation**
  Gemini model generates response using retrieved context

## Setup

**1. Clone the repo**

**2. Create virtual env**

  ```python -m venv venv
  source venv/bin/activate   # Mac/Linux
  venv\Scripts\activate      # Windows
```
  
**3. pip install -r requirements.txt**

**4. Create .env in the root directory. And paste your API key** <br/>
     ```GEMINI_API_KEY=your_api_key_here```

**5. Run the app.**  <br/>
      ```streamlit run app.py```

## Usage ?
- Ask questions related to dehydration
- Enter your question in the input box and click 'Ask' button

## Output

_Enter your question and click "Ask" button ..._
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/bc2a8b65-206d-43ea-b242-0cda8818007a" />

_Response ..._
<img width="710" height="775" alt="image" src="https://github.com/user-attachments/assets/6583423c-3b97-44aa-966c-d18de2449fb8" />

_Other question_
<img width="700" height="550" alt="image" src="https://github.com/user-attachments/assets/ad68a8b1-c941-440d-8749-9af0085ad875" />
