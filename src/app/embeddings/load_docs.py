# src/app/embeddings/load_docs.py
"""
Pipeline para cargar, fragmentar, embeber y almacenar documentos en PGVector.
"""
import os
from typing import List
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import get_settings
from app.embeddings.vectorstore import get_vectorstore


def load_documents(data_dir: str) -> List[Document]:
    """
    Carga archivos .txt y .pdf desde data_dir como Document.
    """
    docs: List[Document] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext == ".txt":
                    docs.extend(TextLoader(path).load())
                elif ext == ".pdf":
                    docs.extend(PyPDFLoader(path).load())
                else:
                    continue
            except Exception as e:
                print(f"Error cargando {path}: {e}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Divide documentos en fragmentos según configuración.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_documents(chunks: List[Document]):
    """
    Genera embeddings y guarda los chunks en PGVector.
    """
    settings = get_settings()
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
    vectorstore = get_vectorstore(
        embedding_function=embeddings.embed_query,
        collection_name=settings.collection_name,
        use_jsonb=True
    )
    vectorstore.add_documents(chunks)


def main():
    settings = get_settings()
    print(f"Cargando documentos de {settings.data_directory}...")
    raw_docs = load_documents(settings.data_directory)
    print(f"Documentos cargados: {len(raw_docs)}")
    chunks = split_documents(raw_docs)
    print(f"Fragmentos creados: {len(chunks)}")
    print("Iniciando ingestión en vectorstore...")
    ingest_documents(chunks)
    print("Ingestión completada.")

if __name__ == "__main__":
    main()
