import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.repositories.qa_repository import QARepository

router = APIRouter(prefix="/qa", tags=["QA Records"])

class QAIn(BaseModel):
    question: str
    answer: str
    metadata: dict
    embedding: List[float] = Field(..., min_items=768, max_items=768)

class RatingIn(BaseModel):
    rating: float = Field(..., ge=1.0, le=10.0)

@router.post("/record", response_model=dict)
async def create_qa(record: QAIn, db: Session = Depends(get_db)):
    repo = QARepository(db)
    qa = repo.create_record(
        question=record.question,
        answer=record.answer,
        metadata=record.metadata,
        embedding=record.embedding
    )
    return {"id": qa.id}

@router.patch("/{qa_id}/rating", response_model=dict)
async def patch_rating(
    qa_id: uuid.UUID,
    payload: RatingIn,
    db: Session = Depends(get_db)
):
    repo = QARepository(db)
    qa = repo.update_rating(qa_id, payload.rating)
    if not qa:
        raise HTTPException(status_code=404, detail="QA record not found")
    return {"id": qa.id, "rating": qa.rating}

@router.get("/positive", response_model=List[dict])
async def get_positive(min_rating: float = 5.0, db: Session = Depends(get_db)):
    repo = QARepository(db)
    records = repo.get_positive_records(min_rating)
    return [{"id": r.id, "rating": r.rating} for r in records]
