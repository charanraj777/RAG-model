from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, query


def create_app() -> FastAPI:
    app = FastAPI(
        title="Insurance Policy RAG API",
        version="1.0.0",
        description=" RAG backend for insurance policy QA.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(query.router, tags=["query"])

    return app


app = create_app()

