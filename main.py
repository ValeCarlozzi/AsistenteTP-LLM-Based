# --- CELDA 6: Configuración principal de LlamaIndex y Groq ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import PDFReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import requests


GROQ_API_KEY = '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'

# Configuramos el LLM
llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
Settings.llm = llm # por defecto usamos el modelo llama3 en los procesos de llamaindex
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# esto vamos a tener que cambiarlo
def getPdf():
    
    import requests

    pdf_url = "https://pdflink.to/efaff9e4/"  # Ley argentina, pública
    pdf_fn = "TP 2 - Agente.pdf"

    r = requests.get(pdf_url)
    with open(pdf_fn, "wb") as f:
        f.write(r.content)

    loader = PDFReader()
    documents = loader.load_data(file=pdf_fn)
    index = VectorStoreIndex.from_documents(documents) 
    return index

# pdf para contexto
index = getPdf()  # <-- corregido, ahora es una llamada a la función

# Armamos prompt
#prompt = "¿Cuáles son las principales diferencias entre Llama 2 y Llama 3?"

# ---- Llama 3 via Groq API ----
def groq_llm(prompt,
             model="llama3-70b-8192", # o llama3-8b-8192 según disponibilidad
             max_tokens=250, temperature=0.1,
             system_message="Sos un asistente experto en IA. Respondé en español."):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": None
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(response.text)
        raise Exception("Error en la API de Groq")


# ----  Ejecutar el LLM sobre prompt ----
prompt ="de que habla este documento?"
respuesta = groq_llm(prompt)
print("\n=== Respuesta Llama 3 (Groq) ===\n")
print(respuesta)

def ask_with_context(question, index, model="llama3-70b-8192"):
    # 1. Recuperar contexto relevante del índice
    retriever = index.as_retriever()
    nodes = retriever.retrieve(question)
    context = "\n".join([node.text for node in nodes])

    # 2. Combinar contexto y pregunta en un prompt
    prompt = (
        "Usá el siguiente contexto extraído de un PDF para responder la pregunta.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )

    # 3. Llamar al LLM con el nuevo prompt
    return groq_llm(prompt, model=model)

# Ejemplo de uso:
question = "de que habla este documento?"
respuesta = ask_with_context(question, index)
print("\n=== Respuesta Llama 3 (Groq) con contexto ===\n")
print(respuesta)