# -*- coding: utf-8 -*-
"""
Pipeline para cargar, fragmentar, embeber y almacenar documentos (TXT, PDF, DOCX) **y**
un CSV ya pre‑segmentado con metadatos (p. ej. el `instructivo_enriched_v2.csv`).

La salida se inserta en PGVector con los metadatos preservados en una columna JSONB.

Cambios principales respecto a la versión original
-------------------------------------------------
1. **Soporte CSV:** se añade `load_csv_chunks()` que transforma cada fila en un
   `langchain_core.documents.Document` listo para embeddings.
2. **Split selectivo:** los documentos que traen `metadata['word_count']` menor
   que `CHUNK_SIZE` se consideran ya fragmentados y no se dividen de nuevo.
3. **Metadatos enriquecidos:** se guardan `chunk_id`, `modulo`, `sistema`,
   `tipo`, `fase`, `word_count`, `doc_version` y `source_file`.

Requisitos extra
----------------
* pandas>=2.0
* python-docx, pillow, pytesseract (ya estaban)
"""

import os
import io
import tempfile
from dotenv import load_dotenv
from typing import List, Dict, Any

import pandas as pd  # Nueva dependencia
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from docx import Document as DocxDocument  # Usar python-docx
from PIL import Image  # Para manejar imágenes
import pytesseract  # Para OCR

# Opcional: para usar un modelo multimodal de Google para captioning
from google.genai import GenerativeModel
from google.genai.types import HarmCategory, HarmBlockThreshold

# ---------------------------------------------------------------------------
# Configuración de entorno
# ---------------------------------------------------------------------------

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "your_db_name")
DB_USER = os.getenv("DB_USER", "your_db_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_db_password")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Necesaria si llamas a la API

COLLECTION_NAME = "knowledge_collection"
EMBEDDING_DIMENSION = 768  # text-embedding-004 devuelve 768‑D

CONNECTION_STRING = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Carpeta donde están los documentos a procesar
DATA_DIRECTORY = "./src/docs"  # <‑‑ Cámbialo si es necesario

# Configuración del Text Splitter (para docs sin segmentar)
CHUNK_SIZE = 400  # 400 tokens ~ ≈300‑350 palabras
CHUNK_OVERLAP = 80

# Configuración de Tesseract OCR (cambia la ruta si es necesario)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# Inicializar modelos
# ---------------------------------------------------------------------------

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("➡️  Modelo de embedding inicializado (text-embedding-004).")
except Exception as e:
    raise SystemExit(f"❌ Error al inicializar embeddings: {e}. Comprueba GOOGLE_API_KEY.")

# Opcional: modelo multimodal para imágenes/captioning
try:
    vision_model = GenerativeModel("models/gemini-2.0-flash-thinking-exp-1219")
    print("➡️  Modelo multimodal (Vision) inicializado.")
except Exception as e:
    print(f"⚠️  Vision no disponible: {e}")
    vision_model = None  # Continuar sin visión

# ---------------------------------------------------------------------------
# Utilidades específicas
# ---------------------------------------------------------------------------

def extract_images_from_docx(path: str):
    """Placeholder – implementa si necesitas extraer imágenes de un DOCX."""
    return []

def process_image(image_bytes: bytes) -> str:
    """Ejecuta OCR o captioning sobre una imagen y devuelve texto descriptivo."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        img = Image.open(tmp_path)
        text = pytesseract.image_to_string(img, lang="spa+eng")
        return text.strip()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Loader para CSV pre‑segmentado
# ---------------------------------------------------------------------------

def load_csv_chunks(path: str) -> List[Document]:
    """Convierte cada fila del CSV enriquecido en un `Document` con metadatos."""
    df = pd.read_csv(path)
    required_cols = {
        "text_enriched",
        "chunk_id",
        "modulo",
        "sistema",
        "tipo",
        "fase",
        "word_count",
        "doc_version",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"El CSV {path} no tiene todas las columnas requeridas: {required_cols}."
        )

    docs: List[Document] = []
    for _, row in df.iterrows():
        metadata: Dict[str, Any] = {
            "chunk_id": row["chunk_id"],
            "modulo": row["modulo"],
            "sistema": row["sistema"],
            "tipo": row["tipo"],
            "fase": row["fase"],
            "word_count": int(row["word_count"]),
            "doc_version": row["doc_version"],
            "source_file": os.path.basename(path),
            "pre_segmented": True,  # bandera para saltar el splitter
        }
        docs.append(Document(page_content=row["text_enriched"], metadata=metadata))
    print(f"  📑 {len(docs)} chunks cargados desde CSV {os.path.basename(path)}.")
    return docs

# ---------------------------------------------------------------------------
# Procesamiento principal
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(DATA_DIRECTORY):
        raise SystemExit(f"❌ El directorio '{DATA_DIRECTORY}' no existe.")

    documents_for_embedding: List[Document] = []
    print(f"🔍 Buscando documentos en '{DATA_DIRECTORY}'…")

    for root, _, files in os.walk(DATA_DIRECTORY):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            ext = os.path.splitext(file_name)[1].lower()
            base_meta = {"source": file_path}

            print(f"\n➡️  Procesando: {file_name}")
            try:
                if ext == ".txt":
                    docs = TextLoader(file_path).load()
                    documents_for_embedding.extend(docs)
                    print(f"  ✔︎ TXT cargado – {len(docs)} páginas.")

                elif ext == ".pdf":
                    docs = PyPDFLoader(file_path).load()
                    documents_for_embedding.extend(docs)
                    print(f"  ✔︎ PDF cargado – {len(docs)} páginas.")

                elif ext == ".docx":
                    docs = process_docx(file_path, base_meta)
                    documents_for_embedding.extend(docs)

                elif ext == ".csv":
                    docs = load_csv_chunks(file_path)
                    documents_for_embedding.extend(docs)

                else:
                    print(f"  ⚠️  Extensión no soportada: {ext}")
            except Exception as e:
                print(f"  ❌ Error procesando {file_name}: {e}")

    if not documents_for_embedding:
        raise SystemExit("❌ No se cargó ningún documento.")

    print(f"\nTotal documentos antes de chunking: {len(documents_for_embedding)}")

    # Seleccionar cuáles necesitan splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: List[Document] = []
    for d in documents_for_embedding:
        if d.metadata.get("pre_segmented") or d.metadata.get("word_count", 0) < CHUNK_SIZE:
            chunks.append(d)
        else:
            chunks.extend(splitter.split_documents([d]))

    print(f"Fragmentos finales a embeber: {len(chunks)}")

    # ---------------------------------------------------------------------
    # PGVector – conexión e inserción
    # ---------------------------------------------------------------------
    try:
        vectorstore = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
            use_jsonb=True,
        )
        print(f"Conectado a PGVector (colección '{COLLECTION_NAME}'). Insertando…")
        vectorstore.add_documents(chunks)
        print("✅ Inserción completada.")
    except Exception as e:
        raise SystemExit(f"❌ Error insertando en PGVector: {e}")

# ---------------------------------------------------------------------------
# Helpers adicionales
# ---------------------------------------------------------------------------

def process_docx(path: str, base_meta: Dict[str, Any]) -> List[Document]:
    """Extrae texto + OCR de imágenes de un DOCX y devuelve un único Document."""
    try:
        parts = extract_images_from_docx(path)  # Implementa esta función si lo necesitas
    except Exception:
        parts = []

    processed_text = ""
    for part in parts:
        if part["type"] == "text":
            processed_text += part["content"] + "\n"
        elif part["type"] == "image":
            ocr = process_image(part["content"])
            if ocr:
                processed_text += f"\n--- Imagen OCR ---\n{ocr}\n--- Fin Imagen OCR ---\n"

    if processed_text.strip():
        return [Document(page_content=processed_text.strip(), metadata=base_meta)]
    else:
        print("  ⚠️  DOCX sin contenido procesable.")
        return []

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
