from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model: str = "mxbai-embed-large:335m" 
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_prefix="EMB_", 
        env_file=".env",
        extra='ignore'
    )
