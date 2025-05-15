# src/app/core/config.py
"""
Configuración centralizada de la aplicación usando Pydantic v2 y pydantic-settings.
Carga variables de entorno desde un archivo .env o el entorno del sistema.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, field_validator
from typing import Optional

class Settings(BaseSettings):

    # --- Database ---
    db_host: str = Field('localhost', env='DB_HOST')
    db_port: int = Field(5432, env='DB_PORT')
    db_name: str = Field('your_db_name', env='DB_NAME')
    db_user: str = Field('your_db_user', env='DB_USER')
    db_password: str = Field('your_db_password', env='DB_PASSWORD')
    database_url: PostgresDsn | None = None

    # --- Embeddings & Vectorstore ---
    collection_name: str = Field('knowledge_collection', env='COLLECTION_NAME')
    embedding_model: str = Field('models/text-embedding-004', env='EMBEDDING_MODEL')
    embedding_dimension: int = Field(768, env='EMBEDDING_DIMENSION')
    llm_model: str = Field('models/gemini-2.0-flash-thinking-exp-1219', env='LLM_MODEL')
    google_api_key: str = Field('your_google_api_key', env='GOOGLE_API_KEY')

    # --- Text splitting ---
    chunk_size: int = Field(1000, env='CHUNK_SIZE')
    chunk_overlap: int = Field(200, env='CHUNK_OVERLAP')

    # --- Documents ---
    data_directory: str = Field('/src/docs', env='DATA_DIRECTORY')

    # — Este campo se construye dinámicamente si no viene en .env —
    database_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @field_validator("database_url", mode="before")
    def assemble_db_connection(cls, v, info):
        # Si ya viene en .env, lo usa; si no, lo arma aquí:
        if v:
            return v
        data = info.data
        return (
            f"postgresql://{data['db_user']}:{data['db_password']}"
            f"@{data['db_host']}:{data['db_port']}/{data['db_name']}"
        )


# Instancia que puedes importar directamente
t_settings = Settings()

def get_settings() -> Settings:
    return t_settings