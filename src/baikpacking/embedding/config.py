from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model: str = "mxbai-embed-large:335m" 
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333" 
    qdrant_api_key: str | None = None
    qdrant_collection: str = "bikepacking_riders_v2"

    model_config = SettingsConfigDict(
        env_prefix="EMB_", 
        env_file=".env",
        extra='ignore'
    )
