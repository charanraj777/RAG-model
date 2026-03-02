from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")
    OPENAI_API_BASE: str = Field("", env="OPENAI_API_BASE")

    LLM_MODEL: str = Field("gpt-4o-mini", env="LLM_MODEL")
    EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")

    # Local mode (no OpenAI key): Ollama LLM + HuggingFace embeddings
    OLLAMA_BASE_URL: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field("llama3.2", env="OLLAMA_MODEL")
    LOCAL_EMBEDDING_MODEL: str = Field(
        "BAAI/bge-small-en-v1.5",
        env="LOCAL_EMBEDDING_MODEL",
    )

    VECTOR_DB_PATH: str = Field("data/vector_store", env="VECTOR_DB_PATH")
    VECTOR_COLLECTION_NAME: str = Field(
        "insurance_policies", env="VECTOR_COLLECTION_NAME"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

