# src/app/database/helpers/record_helper.py
from app.database.session import SessionLocal
from app.database.repositories.qa_repository import QARepository
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import threading, uuid
from typing import Any, Callable, Dict, List
from types import MethodType

def _resolve_embedding_fn(qa_chain):
    """
    Devuelve una función embed_query(text) -> List[float].
    1) Intenta retriever.vectorstore.embedding_function
       • Si ya es callable  -> la usa.
       • Si es instancia de embeddings -> usa .embed_query.
    2) Fallback: crea un encoder Gemini y devuelve .embed_query.
    """
    try:
        emb_attr = qa_chain.retriever.vectorstore.embedding_function
        # — Caso A: ya es función
        if callable(emb_attr):
            return emb_attr
        # — Caso B: es el objeto embeddings (no callable)
        if isinstance(emb_attr, GoogleGenerativeAIEmbeddings):
            return emb_attr.embed_query
        # — Caso C: prueba si trae método embed_query
        if hasattr(emb_attr, "embed_query") and isinstance(
            emb_attr.embed_query, MethodType
        ):
            return emb_attr.embed_query
    except AttributeError:
        pass  # seguimos al fallback

    # Fallback final: crear un encoder nuevo
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    ).embed_query


# ------------------------------------------------------------------
# 1) Inserción síncrona – devuelve UUID
# ------------------------------------------------------------------
def log_qa_record(
    question: str,
    answer: str,
    qa_chain,
    user_id: str | None = None,
) -> uuid.UUID:
    # 1) Obtener embedding de la pregunta
    embedding_fn = _resolve_embedding_fn(qa_chain)
    vector = embedding_fn(question)

    # 2) Invocar la chain para obtener answer y docs

    source_docs = qa_chain.retriever.get_relevant_documents(question)
    chunks = [d.metadata.get("chunk_id") for d in source_docs if d.metadata.get("chunk_id")]

    # 3) Armar metadata y guardar
    metadata: Dict[str, Any] = {
        "chunks": chunks,
        "user_id": user_id,
        "model": "gemini-text-embedding-004",
    }

    db = SessionLocal()
    try:
        repo = QARepository(db)
        record = repo.create_record(
            question=question,
            answer=answer,
            metadata=metadata,
            embedding=vector,
        )
        return record.id
    finally:
        db.close()



# ------------------------------------------------------------------
# 2) Wrapper asíncrono – devuelve UUID o None
# ------------------------------------------------------------------
def log_qa_async(*args, **kwargs):
    result_container: Dict[str, Any] = {}

    def _target():
        try:
            result_container["id"] = log_qa_record(*args, **kwargs)
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()  # bloquea milisegundos para obtener el id
    if "error" in result_container:
        raise result_container["error"]
    return result_container.get("id")


# ------------------------------------------------------------------
# 3) Actualizar rating (síncrono)
# ------------------------------------------------------------------
def update_rating(record_id: uuid.UUID, rating: int):
    db = SessionLocal()
    try:
        repo = QARepository(db)
        repo.update_rating(record_id, rating)
    finally:
        db.close()


# ------------------------------------------------------------------
# 4) Wrapper asíncrono para rating
# ------------------------------------------------------------------
def update_rating_async(record_id: uuid.UUID, rating: int) -> bool:
    result_container: Dict[str, Any] = {}

    def _target():
        try:
            update_rating(record_id, rating)
            result_container["ok"] = True
        except Exception as e:
            result_container["ok"] = False
            result_container["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if not result_container.get("ok"):
        raise result_container.get("error", RuntimeError("Unknown error"))
    return True
