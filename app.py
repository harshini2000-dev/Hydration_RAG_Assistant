import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import numpy as np

# ------------------------
# Load API key
# ------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

st.title("💧 Hydration RAG Assistant")
st.write("Ask questions about dehydration.")

# ------------------------
# Load & Prepare Data (Cached)
# ------------------------
@st.cache_resource
def load_rag_system():

    # Load PDF
    reader = PdfReader("Dehydration.pdf")
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    # Chunk
    def chunk_text(text, chunk_size=150, overlap=30):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    chunks = chunk_text(full_text)

    # Embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ------------------------
    # FAISS: Load or Create
    # ------------------------
    if os.path.exists("faiss.index") and os.path.exists("chunks.npy"):
        # Load saved index + chunks
        index = faiss.read_index("faiss.index")
        chunks = np.load("chunks.npy", allow_pickle=True).tolist()
    else:
        # Create embeddings
        embeddings = embed_model.encode(chunks, convert_to_numpy=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save for future runs
        faiss.write_index(index, "faiss.index")
        np.save("chunks.npy", chunks)

    return chunks, embed_model, index

# call load_rag_system() function
chunks, embed_model, index = load_rag_system()


# ------------------------
# Retrieval Function
# ------------------------
def retrieve_chunks(query, k= 3):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


# ------------------------
# Answer Function
# ------------------------
def answer_question(query):
    context_chunks = retrieve_chunks(query)
    context_text = "\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer the question clearly and accurately. If the question asked is not related to hydration or health then say 'The question is out of context'

Context:
{context_text}

Question:
{query}

Answer:
"""

    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)

    return response.text


# ------------------------
# UI Interaction
# ------------------------
user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_question:
        with st.spinner("Thinking..."):
            answer = answer_question(user_question)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")