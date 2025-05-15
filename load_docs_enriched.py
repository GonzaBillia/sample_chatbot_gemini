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

import os, csv
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

# --- Import nuevo SDK google-genai ---
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuración de entorno
# ---------------------------------------------------------------------------

load_dotenv()

DB_HOST         = os.getenv("DB_HOST", "localhost")
DB_PORT         = os.getenv("DB_PORT", "5432")
DB_NAME         = os.getenv("DB_NAME", "your_db_name")
DB_USER         = os.getenv("DB_USER", "your_db_user")
DB_PASSWORD     = os.getenv("DB_PASSWORD", "your_db_password")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")  # Necesaria para genai.Client

COLLECTION_NAME     = "knowledge_collection"
EMBEDDING_DIMENSION = 768  # text-embedding-004 devuelve 768-D

CONNECTION_STRING = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

DATA_DIRECTORY = "./src/docs"  # Carpeta de documentos

CHUNK_SIZE    = 400  # tokens (~300-350 palabras)
CHUNK_OVERLAP = 80

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# Inicializar modelos
# ---------------------------------------------------------------------------

# Embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("➡️  Modelo de embedding inicializado (text-embedding-004).")
except Exception as e:
    raise SystemExit(f"❌ Error al inicializar embeddings: {e}. Comprueba GOOGLE_API_KEY.")

# Cliente unificado de Gemini (texto y visión)
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("➡️  Cliente google-genai inicializado.")
    vision_model = client  # usa client.generate() para captioning
except Exception as e:
    print(f"⚠️  No se pudo inicializar google-genai: {e}")
    vision_model = None

# ---------------------------------------------------------------------------
# Utilidades específicas
# ---------------------------------------------------------------------------

def extract_images_from_docx(path: str):
    """Placeholder – implementa si necesitas extraer imágenes de un DOCX."""
    return []

def process_image(image_bytes: bytes) -> str:
    """Ejecuta OCR o captioning sobre una imagen y devuelve texto descriptivo."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        img = Image.open(tmp_path)
        # Primero OCR
        text = pytesseract.image_to_string(img, lang="spa+eng").strip()

        # Si hay cliente Vision, añadir captioning de Gemini
        if vision_model:
            # ejemplo de uso; no explora exhaustivamente todos los parámetros
            response = vision_model.generate(
                model="gemini-2.0-flash-thinking-exp-1219",
                prompt="Describe esta imagen de forma concisa."
            )
            caption = response.text or ""
            text = (text + "\n" + caption).strip()

        return text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Loader para CSV pre‑segmentado
# ---------------------------------------------------------------------------

def load_csv_chunks(path: str) -> List[Document]:
    """Convierte cada fila del CSV enriquecido en un `Document` con metadatos,
    detectando automáticamente el delimitador y cargando todas las columnas QA."""
    # 1) Detectar delimitador
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        sep = dialect.delimiter

    # 2) Leer CSV
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    required_cols = {
        "chunk_id",
        "text_enriched",
        "word_count",
        "modulo",
        "sistema",
        "tipo",
        "fase",
        "doc_version",
        "summary",
        "qa_1_question",
        "qa_1_answer",
        "qa_2_question",
        "qa_2_answer",
        "qa_3_question",
        "qa_3_answer",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"El CSV {path} no tiene las columnas requeridas: {', '.join(missing)}."
        )

    docs: List[Document] = []
    for _, row in df.iterrows():
        metadata: Dict[str, Any] = {
            "chunk_id":           row["chunk_id"],
            "word_count":         int(row["word_count"]),
            "modulo":             row["modulo"],
            "sistema":            row["sistema"],
            "tipo":               row["tipo"],
            "fase":               row["fase"],
            "doc_version":        row["doc_version"],
            "summary":            row["summary"],
            # pares pregunta-respuesta
            "qa_1_question":      row["qa_1_question"],
            "qa_1_answer":        row["qa_1_answer"],
            "qa_2_question":      row["qa_2_question"],
            "qa_2_answer":        row["qa_2_answer"],
            "qa_3_question":      row["qa_3_question"],
            "qa_3_answer":        row["qa_3_answer"],
            # datos propios del loader
            "source_file":        os.path.basename(path),
            "pre_segmented":      True,
        }
        docs.append(
            Document(
                page_content=row["text_enriched"],
                metadata=metadata
            )
        )

    print(f"  📑 {len(docs)} chunks cargados desde CSV {os.path.basename(path)} (sep='{sep}').")
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
