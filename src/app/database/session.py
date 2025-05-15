# src/app/db/session.py
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Carga variables de entorno desde .env
t_load = load_dotenv()

# Configuración de conexión
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "your_db_name")
DB_USER     = os.getenv("DB_USER", "your_db_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_db_password")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Motor de base de datos SQLAlchemy
en_ = create_engine(
    DATABASE_URL,
    echo=False,         # True para debug de SQL
    pool_size=10,       # Tamaño inicial del pool
    max_overflow=20,    # Conexiones extra por encima de pool_size
    future=True,        # Usa la API 2.0 de SQLAlchemy
)

SessionLocal = sessionmaker(
    bind=en_,
    autocommit=False,
    autoflush=False,
    future=True,
)

# Declarative base para modelos ORM
Base = declarative_base()

# Dependencia para FastAPI
def get_db():
    """
    Provee una sesión de base de datos a los endpoints de FastAPI.
    Usage:
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
