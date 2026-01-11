@st.cache_resource
def cargar_llm():
    return Ollama(model="llama3", temperature=0.2)

llm = cargar_llm()

@st.cache_resource
def cargar_sistema():
    textos = []
    ruta = "data"

    for archivo in os.listdir(ruta):
        if archivo.lower().endswith(".pdf"):
            reader = PdfReader(os.path.join(ruta, archivo))
            for pagina in reader.pages:
                texto = pagina.extract_text()
                if texto and len(texto) > 50:
                    textos.append(texto)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_text("\n".join(textos))

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return llm, embedder, index, chunks

    def buscar_contexto(pregunta, embedder, index, chunks, k=3):
    vec = embedder.encode([pregunta])
    _, indices = index.search(vec, k)

    contexto = []
    for i in indices[0]:
        if i < len(chunks):
            contexto.append(chunks[i][:400])

    return "\n\n".join(contexto)

    def preguntar(pregunta):
    llm, embedder, index, chunks = cargar_sistema()
    contexto = buscar_contexto(pregunta, embedder, index, chunks)

    prompt = f"""
Eres un profesor de astronomía.
Responde de forma clara y breve (máx 5 líneas).
Usa SOLO el contexto.
Si no hay información suficiente, dilo.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta:
"""

    return llm.invoke(prompt).strip()

st.set_page_config(page_title="Chatbot del Universo", layout="centered")
st.image(
    "images/universo.jpg",
    use_container_width=True
)

st.title("🌌 Chatbot del Universo")
st.write("Pregunta sobre astronomía usando libros en PDF")

pregunta = st.text_input("Escribe tu pregunta:")

if pregunta:
    with st.spinner("Pensando..."):
        respuesta = preguntar(pregunta)

    st.image(
        imagen_por_pregunta(pregunta),
        use_container_width=True
    )

    st.markdown("### Respuesta")
    st.write(respuesta)