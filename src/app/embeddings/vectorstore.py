# src/app/embeddings/vectorstore.py
"""
Wrapper para inicializar y obtener el cliente PGVector configurado.
"""
from langchain_community.vectorstores import PGVector
from app.core.config import get_settings


def get_vectorstore(
    embedding_function,
    collection_name: str | None = None,
    use_jsonb: bool = True,
) -> PGVector:
    """
    Devuelve una instancia de PGVector configurada con la URL de conexión
y el nombre de colección definidos en Settings.

    Args:
        embedding_function: función que recibe texto y devuelve un vector.
        collection_name: opcional, si se quiere usar un nombre distinto al configurado.
        use_jsonb: si se almacenan vectores en JSONB (recomendado para pgvector).

    Returns:
        PGVector: objeto para interactuar con la colección en Postgres.
    """
    settings = get_settings()
    conn_str = settings.database_url
    coll = collection_name or settings.collection_name

    vectorstore = PGVector(
        collection_name=coll,
        connection_string=conn_str,
        embedding_function=embedding_function,
        use_jsonb=use_jsonb,
    )
    return vectorstore
