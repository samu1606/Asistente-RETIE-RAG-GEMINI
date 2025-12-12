import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuración de página
st.set_page_config(page_title="Asistente RETIE", page_icon="⚡")

load_dotenv()

# Título y presentación
st.title("⚡ Asistente Ingeniero RETIE")
st.markdown("""
Soy un asistente especializado en el Reglamento Técnico de Instalaciones Eléctricas.
Pregúntame sobre distancias, calibres, o artículos específicos.
""")

# Función para cargar la DB (cacheada para velocidad)
@st.cache_resource
def load_chain():
    # 1. Configurar Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 2. Cargar la base de datos existente
    vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
    
    # 3. Configurar el LLM (Gemini Pro)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2) # Temp baja para ser preciso
    
    # 4. Prompt de Ingeniería (Instrucciones al sistema)
    template = """Eres un Ingeniero Electricista experto en normativa colombiana (RETIE/NTC 2050).
    Usa los siguientes fragmentos de contexto para responder la pregunta al final.
    Si no sabes la respuesta, di que no está en el documento, no inventes.
    Cita siempre el artículo o tabla si es posible.
    
    Contexto: {context}
    
    Pregunta: {question}
    
    Respuesta Técnica:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 5. Crear cadena de recuperación
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}), # Busca los 5 fragmentos más relevantes
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# Inicializar cadena
chain = load_chain()

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Ej: ¿Cuál es la distancia de seguridad en 13.2 kV?"):
    # Guardar y mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Consultando la norma..."):
            response = chain.invoke({"query": prompt})
            answer = response['result']
            
            st.markdown(answer)
            
            # Opcional: Mostrar fuentes (útil para ingeniería)
            with st.expander("Ver fuentes consultadas"):
                for doc in response['source_documents']:
                    st.caption(f"Fragmento: {doc.page_content[:200]}...")

    # Guardar respuesta
    st.session_state.messages.append({"role": "assistant", "content": answer})