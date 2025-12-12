import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar variables de entorno
load_dotenv()

def ingest_docs():
    pdf_path = r"C:\Users\samue\OneDrive\Documents\ASISTENTES RETIE CON IA\Asistente RETIE RAG GEMINI\NORMATIVA_COLOMBIA\NTC 2050\codigo-electrico-colombiano-segunda-actualizacion-pdf_compress.pdf" # Asegúrate de que el nombre coincida
    
    print("⚡ Cargando PDF... esto puede tardar un poco.")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Dividir el texto en fragmentos (chunks)
    # 1000 caracteres con 100 de solapamiento para no cortar ideas
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    print(f"📄 Se generaron {len(texts)} fragmentos de texto.")

    # Crear Embeddings (convertir texto a vectores) usando Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Guardar en base de datos local (ChromaDB)
    print("💾 Guardando en base de datos vectorial...")
    vector_db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./vector_db"
    )
    
    print("✅ ¡Listo! Base de datos creada en /vector_db")

if __name__ == "__main__":
    ingest_docs()