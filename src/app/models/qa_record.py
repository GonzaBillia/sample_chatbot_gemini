import uuid
from datetime import datetime
from sqlalchemy import Column, Text, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.database.session import Base

class QARecord(Base):
    __tablename__ = "qa_records"
    __table_args__ = {"schema": "qa"}

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question   = Column(Text, nullable=False)
    answer     = Column(Text, nullable=False)
    metadata   = Column(JSON, nullable=False)
    rating     = Column(Float, nullable=True)
    embedding  = Column(Vector(768), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
