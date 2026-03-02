from typing import List
import os
import uuid
import asyncio

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import BackgroundTasks

from app.services.ingestion_service import IngestionService


router = APIRouter()


@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more insurance policy PDFs"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    doc_ids: List[str] = []
    saved_paths: List[str] = []

    os.makedirs("data/raw", exist_ok=True)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. Invalid file: {f.filename}",
            )

        doc_id = str(uuid.uuid4())
        target_path = os.path.join("data", "raw", f"{doc_id}.pdf")
        content = await f.read()
        async with asyncio.Lock():
            with open(target_path, "wb") as out:
                out.write(content)

        doc_ids.append(doc_id)
        saved_paths.append(target_path)

    ingestion_service = IngestionService()

    for doc_id, path in zip(doc_ids, saved_paths):
        background_tasks.add_task(ingestion_service.process_pdf, doc_id, path)

    return {"status": "accepted", "document_ids": doc_ids}

