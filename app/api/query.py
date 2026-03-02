from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.rag_service import RAGService


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    conversation_id: Optional[str] = None


class Source(BaseModel):
    document: str
    page: int
    section: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    conversation_id: Optional[str] = None


@router.post("/query", response_model=QueryResponse)
async def query_policies(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    rag = RAGService()
    result: Dict[str, Any] = await rag.answer_question(
        query=req.query,
        top_k=req.top_k,
        conversation_id=req.conversation_id,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[
            Source(
                document=src.get("document_name", src.get("source", "unknown")),
                page=src.get("page", -1),
                section=src.get("section"),
                score=src.get("score"),
            )
            for src in result.get("sources", [])
        ],
        conversation_id=result.get("conversation_id"),
    )

