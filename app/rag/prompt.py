from typing import List, Optional


def build_grounded_prompt(
    query: str,
    contexts: List[str],
    history: Optional[List[str]] = None,
) -> str:
    history = history or []

    context_block = "\n\n".join(contexts)
    history_block = "\n".join(history)

    instructions = """
You are an expert assistant that answers questions about insurance policy documents.

You MUST strictly follow these rules:
- Answer ONLY based on the context sections below.
- If the answer is not clearly contained in the context, reply exactly with: "Not available in document".
- Do NOT use any external knowledge or make assumptions.
- Always include citations in your answer. Cite as [Document X, Page Y, Section Z] based on the source snippets.

Format your answer as a clear explanation followed by a "Citations" section listing the sources you used.
"""

    prompt = f"""{instructions}

Previous conversation (may be empty):
{history_block}

Context:
{context_block}

User question:
{query}

Remember: If the answer is not clearly in the context, you MUST answer exactly: "Not available in document".
"""

    return prompt.strip()

