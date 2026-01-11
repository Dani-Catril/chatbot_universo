import streamlit as st
import os
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama


# --------------------------------------------------
# CONFIG STREAMLIT
# --------------------------------------------------
st.set_page_config(
    page_title="Chatbot del Universo",
    page_icon="🌌",
    layout="centered"
)


# --------------------------------------------------
# CARGA DEL SISTEMA (CACHEADA)
# --------------------------------------------------
@st.cache_resource
def cargar_sistema():
    textos = []

    for archivo in os.listdir("data"):
        if archivo.lower().endswith(".pdf"):
            ruta = os.path.join("data", archivo)
            try:
                reader = PdfReader(ruta)
                for pagina in reader.pages:
                    try:
                        texto = pagina.extract_text()
                        if texto and len(texto) > 50:
                            textos.append(texto)
                    except:
                        continue
            except:
                continue

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_text("\n".join(textos))

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    llm = Ollama(
        model="llama3",
        temperature=0.2
    )

    return llm, embedder, index, chunks


# --------------------------------------------------
# BÚSQUEDA DE CONTEXTO
# --------------------------------------------------
def buscar_contexto(pregunta, embedder, index, chunks, k=3):
    pregunta_vec = embedder.encode([pregunta])
    _, indices = index.search(pregunta_vec, k)

    contexto = []
    for i in indices[0]:
        if i < len(chunks):
            texto = chunks[i].strip()
            if len(texto) > 80:
                contexto.append(texto[:300])

    return "\n\n".join(contexto)


# --------------------------------------------------
# PREGUNTAR AL MODELO
# --------------------------------------------------
def preguntar(pregunta, llm, embedder, index, chunks):
    contexto = buscar_contexto(pregunta, embedder, index, chunks)

    prompt = f"""
Eres un profesor de astronomía.
Responde SOLO usando la información del contexto.
Máximo 4 líneas.
Si no hay información suficiente, dilo explícitamente.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta clara y directa:
"""
    return llm.invoke(prompt).strip()


# --------------------------------------------------
# INTERFAZ STREAMLIT
# --------------------------------------------------
st.title("🌌 Chatbot del Universo")
st.write("Preguntas basadas exclusivamente en documentos de astronomía.")

with st.spinner("Cargando sistema..."):
    llm, embedder, index, chunks = cargar_sistema()

pregunta = st.text_input("Haz una pregunta sobre el universo:")

if pregunta:
    with st.spinner("Pensando..."):
        respuesta = preguntar(pregunta, llm, embedder, index, chunks)

    st.markdown("### 📘 Respuesta")
    st.write(respuesta)

