from typing import Optional

from app.utils.settings import settings


def _use_openai() -> bool:
    return bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip())


async def _generate_openai(prompt: str, temperature: float) -> str:
    import openai

    openai.api_key = settings.OPENAI_API_KEY
    if settings.OPENAI_API_BASE:
        openai.api_base = settings.OPENAI_API_BASE
    response = await openai.ChatCompletion.acreate(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an insurance policy assistant. You must answer ONLY using the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"].strip()


async def _generate_ollama(prompt: str, temperature: float) -> Optional[str]:
    import httpx

    system = "You are an insurance policy assistant. You must answer ONLY using the provided context."
    base = settings.OLLAMA_BASE_URL.rstrip("/")
    try:
        # Use a generous timeout; first generation after a pull can be slow.
        async with httpx.AsyncClient(timeout=300.0) as client:
            # 1) Try Ollama "generate" endpoint (older + common)
            url = f"{base}/api/generate"
            payload = {
                "model": settings.OLLAMA_MODEL,
                "prompt": f"{system}\n\n{prompt}",
                "stream": False,
                "options": {"temperature": temperature},
            }
            r = await client.post(url, json=payload)
            if r.status_code == 404:
                # 2) Try Ollama "chat" endpoint
                url = f"{base}/api/chat"
                payload = {
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": temperature},
                }
                r = await client.post(url, json=payload)

            if r.status_code == 404:
                # 3) Try OpenAI-compatible endpoint (some servers)
                url = f"{base}/v1/chat/completions"
                payload = {
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "stream": False,
                }
                r = await client.post(url, json=payload)

            if r.status_code == 404:
                # Not an Ollama-compatible server at this base URL (or Ollama not exposing APIs)
                return None

            r.raise_for_status()
            data = r.json()

        if "response" in data:
            return (data.get("response") or "").strip()
        if "message" in data and isinstance(data["message"], dict):
            return (data["message"].get("content") or "").strip()
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message") or {}
            return (msg.get("content") or "").strip()
        return None
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError):
        # Ollama not running or not installed — caller can fall back to context-only
        return None


class LLMClient:
    async def generate(self, prompt: str, *, temperature: float = 0.0) -> Optional[str]:
        if _use_openai():
            return await _generate_openai(prompt, temperature)
        return await _generate_ollama(prompt, temperature)

