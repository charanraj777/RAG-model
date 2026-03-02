import os
from typing import List, Dict, Any

from app.pdf_processing.pdf_reader import PDFReader
from app.pdf_processing.text_chunker import TextChunker
from app.embeddings.embedding_store import EmbeddingStore
from app.models.schemas import ChunkMetadata


class IngestionService:
    def __init__(self) -> None:
        self.reader = PDFReader()
        self.chunker = TextChunker()
        self.store = EmbeddingStore()

    def process_pdf(self, document_id: str, pdf_path: str) -> None:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        pages = self.reader.extract_pages(pdf_path)

        chunks: List[Dict[str, Any]] = self.chunker.chunk_document(
            document_id=document_id,
            pdf_path=pdf_path,
            pages=pages,
        )

        metadatas: List[ChunkMetadata] = []
        texts: List[str] = []

        for c in chunks:
            texts.append(c["text"])
            metadatas.append(
                ChunkMetadata(
                    document_id=document_id,
                    document_name=os.path.basename(pdf_path),
                    page=c["page"],
                    section=c.get("section"),
                    chunk_id=c["chunk_id"],
                    pdf_path=pdf_path,
                )
            )

        os.makedirs("data/processed", exist_ok=True)
        raw_txt_path = os.path.join("data", "processed", f"{document_id}.txt")
        with open(raw_txt_path, "w", encoding="utf-8") as f:
            for p in pages:
                f.write(f"--- Page {p.page_number} ---\n")
                f.write(p.text)
                f.write("\n\n")

        self.store.add_texts(texts=texts, metadatas=[m.dict() for m in metadatas])

