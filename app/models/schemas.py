from typing import Optional, List

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    document_id: str
    document_name: str
    page: int
    section: Optional[str]
    chunk_id: str
    pdf_path: str


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    pages: int


class ExampleQuery(BaseModel):
    question: str
    description: Optional[str]
    expected_behavior: Optional[str]


class HealthStatus(BaseModel):
    status: str
    vector_store_ready: bool
    llm_ready: bool
    examples: List[ExampleQuery]

