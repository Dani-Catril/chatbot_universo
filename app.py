import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

@st.cache_resource
def cargar_modelo():
    textos = []
    for archivo in os.listdir("data"):
        if archivo.endswith(".pdf"):
            reader = PdfReader(os.path.join("data", archivo))
            for pagina in reader.pages:
                textos.append(pagina.extract_text())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(" ".join(textos))

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = modelo.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return modelo, index, chunks

modelo, index, chunks = cargar_modelo()

def preguntar(pregunta, k=3):
    vector = modelo.encode([pregunta])
    _, indices = index.search(vector, k)
    return " ".join([chunks[i] for i in indices[0]])

st.title("🌌 Chatbot del Universo")
pregunta = st.text_input("Haz una pregunta sobre el universo")

if pregunta:
    st.write("### Respuesta")
    st.write(preguntar(pregunta))