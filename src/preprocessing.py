def dividir_texto(textos, size=400):
    chunks = []
    for texto in textos:
        palabras = texto.split()
        for i in range(0, len(palabras), size):
            chunks.append(" ".join(palabras[i:i+size]))
    return chunks