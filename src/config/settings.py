import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application-wide settings, credentials, and model configurations"""

    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "botany-index")
    AZURE_SEARCH_SEMANTIC_CONFIG: str = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "botany-semantic-config")

    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "flower_yolov8.pt")
    CLASSIFIER_MODEL_PATH: str = os.getenv("CLASSIFIER_MODEL_PATH", "best_plant_classifier.pt")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    PROVER_TIMEOUT_SECONDS: int = int(os.getenv("PROVER_TIMEOUT_SECONDS", "5"))

settings = Settings()
