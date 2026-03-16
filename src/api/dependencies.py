from functools import lru_cache
from loguru import logger


@lru_cache(maxsize=2)
def get_model(model_type: str = "baseline"):
    """Charge chaque type de modèle une seule fois (cache par model_type).

    maxsize=2 pour conserver baseline et distilbert simultanément en mémoire.
    maxsize=1 évince le premier modèle dès qu'un second est demandé, ce qui
    provoquerait des rechargements coûteux selon l'ordre des requêtes.
    """
    from src.training.predict import load_model
    logger.info(f"Chargement modèle : {model_type}")
    return load_model(model_type)
