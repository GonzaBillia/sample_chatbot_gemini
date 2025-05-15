import os
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # Para dividir texto largo

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# --- Configuración ---
# Google API Key (LangChain_google_genai la busca automáticamente en GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") # Alternativa si no la coge auto

# Configuración de la base de datos
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "your_db_name")
DB_USER = os.getenv("DB_USER", "your_db_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_db_password")
TABLE_NAME_PGVECTOR = "langchain_pg_embedding" # <<-- Usar el nombre de tabla esperado por LangChain
COLLECTION_NAME = "knowledge_collection"     # Nombre para identificar esta colección

EMBEDDING_DIMENSION = 768 # Dimensión para text-embedding-004

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Modelo de Embedding de Google Gemini con LangChain
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("Modelo de Embedding inicializado.")
except Exception as e:
    print(f"Error al inicializar modelo de embedding: {e}")
    print("Asegúrate de que GOOGLE_API_KEY está configurada.")
    exit()

# Reemplaza esto con la lógica para cargar tus propios datos
# Cada item es un Document o se puede convertir a uno
# LangChain espera una lista de Document objetos
raw_knowledge_data = [
    {"content": "El pingüino emperador es la especie de pingüino más alta y pesada y es endémico de la Antártida.", "source": "Wikipedia - Pingüino Emperador"},
    {"content": "Los pingüinos emperador se reproducen en las colonias durante el invierno antártico, viajando hasta 120 km sobre el hielo para llegar a los sitios de reproducción.", "source": "Wikipedia - Pingüino Emperador"},
    {"content": "La fotosíntesis es el proceso por el cual las plantas, las algas y algunas bacterias convierten la energía luminosa en energía química.", "source": "Wikipedia - Fotosíntesis"},
    {"content": "Durante la fotosíntesis en las plantas, la energía lumínica es capturada por la clorofila en los cloroplastos.", "source": "Wikipedia - Fotosíntesis"},
    {"content": "El primer ordenador electrónico digital fue el ENIAC, desarrollado en 1945.", "source": "Historia de la Computación"},
    {"content": "Python es un lenguaje de programación interpretado de alto nivel, creado por Guido van Rossum.", "source": "Wikipedia - Python"},
    {"content": "La capital de Francia es París, conocida por la Torre Eiffel y el Museo del Louvre.", "source": "Geografía"},
    {"content": "El agua (H₂O) es una molécula polar esencial para la vida tal como la conocemos.", "source": "Química Básica"},
    {"content": "La velocidad de la luz en el vacío es de aproximadamente 299,792,458 metros por segundo.", "source": "Física"},
    # Ejemplo de texto más largo que podría necesitar chunking
    {"content": "La historia de la computación es una fascinante narrativa de innovación humana que abarca siglos, desde las herramientas de conteo más primitivas hasta las sofisticadas máquinas cuánticas de hoy. Uno de los hitos tempranos fue el ábaco, una herramienta ancestral utilizada para realizar cálculos aritméticos. Mucho más tarde, en el siglo XVII, inventores como Blaise Pascal y Gottfried Wilhelm Leibniz desarrollaron las primeras calculadoras mecánicas capaces de realizar sumas, restas y, en algunos casos, multiplicaciones y divisiones. El siglo XIX trajo avances significativos con figuras como Charles Babbage, a menudo considerado el 'padre de la computación', quien diseñó la Máquina Analítica, un dispositivo mecánico programable que contenía conceptos clave como la unidad aritmético-lógica, el control de flujo y la memoria. Aunque la máquina de Babbage nunca se completó completamente en su tiempo debido a limitaciones tecnológicas y financieras, sus ideas sentaron las bases para futuros desarrollos. Ada Lovelace, trabajando con Babbage, escribió lo que se considera el primer algoritmo diseñado para ser procesado por una máquina, convirtiéndose en la primera programadora de la historia. La invención del relé y, posteriormente, el tubo de vacío en el siglo XX, allanaron el camino para la computación electrónica. La Segunda Guerra Mundial impulsó el desarrollo de máquinas computacionales más rápidas y complejas para tareas como el descifrado de códigos. El ENIAC, terminado en 1945, es a menudo citado como el primer ordenador electrónico digital de propósito general, aunque era enorme y difícil de reprogramar. La invención del transistor en 1947 por Bell Labs revolucionó la electrónica, permitiendo la creación de ordenadores más pequeños, fiables y eficientes. Esto llevó a la segunda generación de ordenadores en la década de 1950. La invención del circuito integrado (chip) en la década de 1960 marcó el comienzo de la tercera generación, permitiendo empaquetar miles de transistores en un solo chip. La invención del microprocesador en la década de 1970 (cuarta generación) puso una unidad de procesamiento completa en un solo chip, lo que llevó al desarrollo de los ordenadores personales y la revolución de la computación de escritorio. La quinta generación, que continúa hoy, se centra en la inteligencia artificial, el procesamiento paralelo, las interfaces naturales y las redes de computadoras a gran escala como Internet. La miniaturización sigue avanzando a pasos agigantados, y campos emergentes como la computación cuántica prometen capacidades computacionales sin precedentes para resolver problemas actualmente intratables. Desde humildes comienzos con herramientas de conteo hasta la era de la inteligencia artificial y la computación cuántica, la historia de la computación es un testimonio del ingenio humano y su búsqueda incesante de herramientas para procesar información y resolver problemas complejos.", "source": "Historia Extendida de la Computación"},
]

# Convertir datos crudos a Documentos de LangChain
# page_content = content, metadata = {'source': source}
documents = [
    Document(page_content=item["content"], metadata={"source": item.get("source")})
    for item in raw_knowledge_data
]

# --- Dividir documentos en fragmentos (Chunking) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Dividiendo {len(documents)} documentos en fragmentos...")
chunks = text_splitter.split_documents(documents)
print(f"Creados {len(chunks)} fragmentos.")


# --- Almacenar en PGVector ---
print(f"Conectando a la base de datos {DB_NAME} e insertando embeddings en tabla '{TABLE_NAME_PGVECTOR}'...")

try:
    # Usar PGVector.from_documents. Asegúrate de que la tabla TABLE_NAME_PGVECTOR existe
    # Y tiene la estructura esperada por LangChain (uuid, collection_id, embedding, document, cmetadata)
    # Pasarle table_name=TABLE_NAME_PGVECTOR ayuda a asegurar que se usa la tabla correcta
    # con la estructura esperada.
    # vectorstore = PGVector.from_documents(
    #     embedding=embeddings,
    #     documents=chunks,
    #     collection_name=COLLECTION_NAME,
    #     connection_string=CONNECTION_STRING,
    #     table_name=TABLE_NAME_PGVECTOR, # <-- Especificar el nombre de tabla que debe usar (con estructura LangChain)
    #     # Si quieres vaciar la colección antes de insertar:
    #     # pre_delete_collection=True
    # )
    print(f"Embeddings de {len(chunks)} fragmentos insertados correctamente.")

except Exception as e:
    print(f"Error al insertar embeddings en PGVector: {e}")
    print(f"Asegúrate de que la base de datos está corriendo, la extensión 'vector' está habilitada,")
    print(f"la tabla '{TABLE_NAME_PGVECTOR}' existe y tiene la estructura exacta de langchain_pg_embedding.")
    print("Verifica tus credenciales en el archivo .env")