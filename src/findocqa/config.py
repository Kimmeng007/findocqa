from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    google_api_key: str = ""
    mongodb_uri: str = ""
    sec_edgar_user_agent: str = "FinDocQA research project (unset-contact@example.com)"

    mongodb_db_name: str = "findocqa"
    generation_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "BAAI/bge-small-en-v1.5"


settings = Settings()
