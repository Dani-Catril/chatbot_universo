import os
import re
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# Configuración OpenAI
# -----------------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]  # Guarda tu key en Streamlit Secrets
client = OpenAI(api_key=openai_api_key)

# -----------------------------
# Funciones de carga y preprocesamiento
# -----------------------------
@st.cache_resource
def cargar_pdfs(ruta="data"):
    textos = []
    for archivo in os.listdir(ruta):
        if archivo.lower().endswith(".pdf"):
            reader = PdfReader(os.path.join(ruta, archivo))
            for pagina in reader.pages:
                texto = pagina.extract_text()
                if texto and len(texto) > 50:
                    textos.append(texto)
    return textos

@st.cache_resource
def crear_embeddings(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, index

def split_text(texto, chunk_size=500, overlap=80):
    textos = []
    start = 0
    while start < len(texto):
        end = start + chunk_size
        textos.append(texto[start:end])
        start += chunk_size - overlap
    return textos

# -----------------------------
# Funciones de búsqueda y QA
# -----------------------------
def buscar_contexto(pregunta, embedder, index, chunks, k=3):
    pregunta_vec = embedder.encode([pregunta])
    distancias, indices = index.search(pregunta_vec, k)
    contexto = []
    for i in indices[0]:
        if i < len(chunks):
            txt = chunks[i].strip()
            if len(txt) > 50:
                contexto.append(txt[:400])
    return "\n\n".join(contexto)

def preguntar(pregunta, embedder, index, chunks):
    contexto = buscar_contexto(pregunta, embedder, index, chunks)
    prompt = f"""
Eres un profesor de astronomía.
Responde SOLO con la información del contexto.
Máximo 5 líneas.
Si no hay información suficiente, dilo explícitamente.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta clara y directa:
"""
    respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return respuesta.choices[0].message.content.strip()

# -----------------------------
# Función para generar imágenes
# -----------------------------
def imagen_por_pregunta(pregunta):
    # Ejemplo: genera URL de DALL·E (requiere habilitar imagen en OpenAI)
    imagen = client.images.generate(
        model="gpt-image-1",
        prompt=pregunta,
        size="512x512"
    )
    return imagen.data[0].url

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🌌 Chatbot Universo")

# Carga PDFs y crea embeddings (solo la primera vez)
with st.spinner("Cargando PDFs y generando embeddings..."):
    textos = cargar_pdfs()
    all_text = "\n".join(textos)
    chunks = split_text(all_text)
    embedder, index = crear_embeddings(chunks)

# Input del usuario
pregunta = st.text_input("Escribe tu pregunta sobre astronomía:")

if pregunta:
    with st.spinner("Pensando..."):
        respuesta = preguntar(pregunta, embedder, index, chunks)
    
    # Imagen relacionada
    try:
        st.image(imagen_por_pregunta(pregunta), use_container_width=True)
    except Exception:
        st.write("No se pudo generar la imagen.")

    st.markdown("### Respuesta")
    st.write(respuesta)