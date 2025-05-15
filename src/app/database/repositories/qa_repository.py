from typing import List, Optional
import uuid
from sqlalchemy.orm import Session
from app.models.qa_record import QARecord

class QARepository:
    def __init__(self, db: Session):
        self.db = db

    # ---------- CREATE ----------
    def create_record(
        self,
        question: str,
        answer: str,
        metadata: dict,
        embedding: List[float],
    ) -> QARecord:
        record = QARecord(
            question=question,
            answer=answer,
            meta=metadata,
            embedding=embedding,
        )
        self.db.add(record)
        self.db.commit()     # 1) guarda
        self.db.refresh(record)
        return record

    # ---------- UPDATE RATING ----------
    def update_rating(
        self,
        record_id: uuid.UUID,
        rating: float,
    ) -> Optional[QARecord]:
        record = self.db.get(QARecord, record_id)  # Session.get() es la API 2.0
        if not record:
            return None
        record.rating = rating
        self.db.commit()
        self.db.refresh(record)
        return record

    # ---------- LISTA DE POSITIVOS ----------
    def get_positive_records(self, min_rating: float = 5.0) -> List[QARecord]:
        return (
            self.db.query(QARecord)
            .filter(QARecord.rating >= min_rating)
            .all()
        )
