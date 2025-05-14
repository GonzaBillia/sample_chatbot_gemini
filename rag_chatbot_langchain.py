import os
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# --- Configuración ---
# Google API Key ya se busca en GOOGLE_API_KEY por defecto
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Configuración de la base de datos
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "your_db_name")
DB_USER = os.getenv("DB_USER", "your_db_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_db_password")

# **Importante:** Usar el nombre de tabla por defecto de LangChain PGVector
TABLE_NAME_PGVECTOR = "langchain_pg_embedding" # <<-- Debe coincidir con el script de inserción
COLLECTION_NAME = "knowledge_collection"     # Debe coincidir con el script de inserción

EMBEDDING_DIMENSION = 768 # Dimensión para text-embedding-004

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Inicializar Modelos y Vector Store ---

# Modelo de Embedding
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("Modelo de Embedding inicializado.")
except Exception as e:
    print(f"Error al inicializar modelo de embedding: {e}")
    print("Asegúrate de que GOOGLE_API_KEY está configurada.")
    exit()

# Vector Store (Base de Datos)
try:
    # Conectarse a la base de datos PGVector existente
    # Asegúrate de que la tabla TABLE_NAME_PGVECTOR existe y tiene la estructura de LangChain
    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings, # Pasar la función de embedding
        use_jsonb=True # Opcional, si guardas metadata compleja en jsonb
    )
    print(f"Conexión a PGVector '{TABLE_NAME_PGVECTOR}' establecida para colección '{COLLECTION_NAME}'.")
except Exception as e:
    print(f"Error al conectar a PGVector: {e}")
    print(f"Asegúrate de que la base de datos está corriendo, y las tablas '{TABLE_NAME_PGVECTOR}' y 'langchain_pg_collection' existen con la estructura esperada.")
    print("Verifica tus credenciales en el archivo .env")
    exit()


# Crear un Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # k=5 significa recuperar los 5 documentos más similares

# Modelo de Generación (LLM)
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-thinking-exp-1219")
    print("Modelo de Generación (Gemini) inicializado.")
except Exception as e:
     print(f"Error al inicializar modelo de generación: {e}")
     print("Asegúrate de que GOOGLE_API_KEY está configurada y tienes acceso a 'gemini-pro'.")
     exit()

# --- Construir la Cadena RAG ---

PROMPT = PromptTemplate(
    template="""
    Eres un asistente experto en documentación técnica de la empresa Farmacias Sanchez Antoniolli, el cual tiene conocimiento exhaustivo en todos sus servicios y aplicaciones adquiridos (PLEX, Quantio, Humand, POSManager, etc.).
    Sigue estas reglas:
    1. Habla en español claro y profesional.
    2. Responde de forma clara y concisa.
    3. Máximo 400 palabras.
    4. Si no encuentras la respuesta en el contexto, escribe exactamente: “No tengo suficiente información, por favor consulta la documentación.”
    5. Al inicio evita aclaraciones como "Segun la informacion provista" o "Segun el contexto dado" ya que se sobre entiende.

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    # return_source_documents=True, # Descomentar si quieres ver los documentos fuente usados
    chain_type_kwargs={"prompt": PROMPT}
)

print("\n¡Hola! Soy tu chatbot RAG. Hazme una pregunta (escribe 'salir' para terminar).")

# --- Bucle principal del Chatbot ---
while True:
    user_query = input("\nTú: ")

    if user_query.lower() == 'salir':
        print("Adiós!")
        break

    if not user_query.strip():
        print("Por favor, introduce una pregunta.")
        continue

    # Ejecutar la cadena RAG
    try:
        print("Procesando pregunta...")
        result = qa_chain.invoke({"query": user_query})
        bot_response = result.get("result", "Lo siento, no pude obtener una respuesta.")

        # Opcional: Si return_source_documents=True
        # source_documents = result.get("source_documents", [])
        # print("\nFuentes utilizadas:")
        # for doc in source_documents:
        #     print(f"- Source: {doc.metadata.get('source', 'N/A')}, Content: {doc.page_content[:100]}...")


    except Exception as e:
        print(f"Ocurrió un error al procesar la pregunta: {e}")
        bot_response = "Lo siento, hubo un error interno."

    # Mostrar la respuesta
    print(f"\nChatbot: {bot_response}")