import uvicorn
from fastapi import FastAPI

from app.api.v1.endpoints.qa import router as qa_router

app = FastAPI(title="QA Vector Service")

# Montar routers
app.include_router(qa_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
