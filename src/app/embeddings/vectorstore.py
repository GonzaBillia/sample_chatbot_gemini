# src/app/embeddings/vectorstore.py
from langchain_community.vectorstores import PGVector
from app.core.config import get_settings

def get_vectorstore(
    embedding_function,
    collection_name: str | None = None,
) -> PGVector:
    settings = get_settings()
    uri = settings.database_url
    table = collection_name or settings.collection_name

    vectorstore = PGVector(
        connection_string=settings.database_url,                # ← nombre correcto del parámetro
        embedding_function=embedding_function,      # ← nombre correcto para la función de embedding
        collection_name=table,                  # ← tabla donde se guardan los vectores
        use_jsonb=True
    )
    return vectorstore
