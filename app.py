import streamlit as st
import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

@st.cache_resource
def cargar_modelo():
    textos = []

    for pdf in os.listdir("data"):
        if not pdf.lower().endswith(".pdf"):
            continue

        ruta = os.path.join("data", pdf)

        try:
            reader = PdfReader(ruta)
        except Exception as e:
            st.warning(f"No se pudo abrir {pdf}")
            continue

        for pagina in reader.pages:
            try:
                texto = pagina.extract_text()
                if texto and len(texto.strip()) > 50:
                    textos.append(texto)
            except Exception:
                # ignora páginas con fuentes rotas
                continue

    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = modelo.encode(textos, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    return modelo, index, textos

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

