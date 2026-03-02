import re
import uuid
from typing import Any, Dict, List

from app.pdf_processing.pdf_reader import PDFPage


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _guess_sections(self, text: str) -> List[str]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        sections: List[str] = []
        current_section = "General"
        sections.append(current_section)

        for line in lines:
            if (
                line.isupper()
                or line.endswith(":")
                or re.match(r"^[0-9.]+\s+[A-Z].+", line)
            ):
                current_section = line
            sections.append(current_section)

        if not sections:
            sections.append("General")
        return sections

    def chunk_document(
        self,
        document_id: str,
        pdf_path: str,
        pages: List[PDFPage],
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for page in pages:
            text = page.text
            if not text.strip():
                continue

            tokens = re.split(r"(\s+)", text)
            section_hints = self._guess_sections(text)

            current: List[str] = []
            current_len = 0
            section = section_hints[0] if section_hints else "General"

            for idx, token in enumerate(tokens):
                current.append(token)
                current_len += len(token)

                if idx < len(section_hints):
                    section = section_hints[idx] or section

                if current_len >= self.chunk_size:
                    chunk_text = "".join(current).strip()
                    if chunk_text:
                        chunks.append(
                            {
                                "chunk_id": str(uuid.uuid4()),
                                "text": chunk_text,
                                "page": page.page_number,
                                "section": section[:200],
                            }
                        )

                    if self.chunk_overlap > 0:
                        overlap_chars = self.chunk_overlap
                        joined = "".join(current)
                        overlap_text = joined[-overlap_chars:]
                        current = [overlap_text]
                        current_len = len(overlap_text)
                    else:
                        current = []
                        current_len = 0

            if current:
                chunk_text = "".join(current).strip()
                if chunk_text:
                    chunks.append(
                        {
                            "chunk_id": str(uuid.uuid4()),
                            "text": chunk_text,
                            "page": page.page_number,
                            "section": section[:200],
                        }
                    )

        return chunks

