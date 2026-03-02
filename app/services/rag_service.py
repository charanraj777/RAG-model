from typing import Any, Dict, List, Optional
import asyncio

from app.embeddings.embedding_store import EmbeddingStore
from app.rag.llm_client import LLMClient
from app.rag.prompt import build_grounded_prompt
from app.utils.memory import ConversationMemory


class RAGService:
    def __init__(self) -> None:
        self.store = EmbeddingStore()
        self.llm = LLMClient()
        self.memory = ConversationMemory()

    async def answer_question(
        self,
        query: str,
        top_k: int = 5,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if conversation_id is None:
            conversation_id = self.memory.new_conversation()

        retrieval_task = asyncio.to_thread(
            self.store.similarity_search_with_score,
            query,
            top_k,
        )
        results = await retrieval_task

        contexts: List[str] = []
        sources: List[Dict[str, Any]] = []

        for doc, score in results:
            meta = doc.metadata or {}
            text = doc.page_content
            contexts.append(
                f"[Document: {meta.get('document_name', 'unknown')} | "
                f"Page: {meta.get('page', 'N/A')} | "
                f"Section: {meta.get('section', 'N/A')}]\n{text}"
            )
            sources.append(
                {
                    "document_name": meta.get("document_name", "unknown"),
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "score": float(score),
                }
            )

        history = self.memory.get_history(conversation_id)

        prompt = build_grounded_prompt(
            query=query,
            contexts=contexts,
            history=history,
        )

        answer = await self.llm.generate(prompt)

        # When no LLM is available (e.g. Ollama not running), return retrieved chunks as the answer
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            if contexts:
                answer = (
                    "Relevant passages from the documents (no LLM available — start Ollama or set OPENAI_API_KEY for a summarized answer):\n\n"
                    + "\n\n---\n\n".join(contexts)
                )
            else:
                answer = "No relevant passages found in the documents. Upload PDFs first, or try a different query."

        self.memory.append_turn(
            conversation_id=conversation_id,
            user=query,
            assistant=answer,
        )

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id,
        }

