import sys
from pathlib import Path
from loguru import logger
from src.common.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> - {message}",
    )
    # Créer le répertoire avant d'ajouter le sink fichier :
    # loguru lève FileNotFoundError si logs/ n'existe pas encore.
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.add(
        "logs/app.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )
