from dataclasses import dataclass
from typing import List

import pdfplumber


@dataclass
class PDFPage:
    page_number: int
    text: str


class PDFReader:
    def extract_pages(self, pdf_path: str) -> List[PDFPage]:
        pages: List[PDFPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(PDFPage(page_number=i + 1, text=text))
        return pages

