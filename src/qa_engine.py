from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("all-MiniLM-L6-v2")

def responder(pregunta, textos, index, k=3):
    emb = modelo.encode([pregunta])
    _, indices = index.search(emb, k)
    return "\n\n".join([textos[i] for i in indices[0]])