import os
import pickle
import time
from utils.logger import get_logger

logger = get_logger("Cache")


def is_cache_expired(path: str, max_age_seconds: int) -> bool:
    """
    Vérifie si le cache est expiré.

    Args:
        path (str): Chemin du fichier cache.
        max_age_seconds (int): Durée maximale en secondes avant expiration.

    Returns:
        bool: True si le cache est expiré ou inexistant, False sinon.
    """
    if not os.path.exists(path):
        return True
    if max_age_seconds is None:
        return False
    age = time.time() - os.path.getmtime(path)
    if age > max_age_seconds:
        logger.info(f"[CACHE EXPIRED] {path} (age: {age:.0f}s)")
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to delete expired cache {path}: {e}")
        return True
    return False


def is_cache_nearly_expired(
    path: str, max_age_seconds: int, threshold: float = 0.8
) -> bool:
    """
    Vérifie si le cache est valide mais proche de l'expiration.

    Utile pour déclencher un refresh proactif avant que le cache
    ne devienne invalide (stratégie "refresh before stale").

    Args:
        path (str): Chemin du fichier cache.
        max_age_seconds (int): Durée maximale en secondes avant expiration.
        threshold (float): Seuil de déclenchement (défaut 0.8 = 80% du TTL écoulé).

    Returns:
        bool: True si le cache est proche de l'expiration, False sinon.
    """
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age > (max_age_seconds * threshold)


def load_cache_file(path: str):
    """
    Charge un objet depuis le cache sans vérifier l'âge.

    Args:
        path (str): Chemin du fichier cache.

    Returns:
        Any: Objet chargé ou None si erreur.
    """
    try:
        with open(path, "rb") as f:
            logger.info(f"[CACHE HIT] Loading from {path}")
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load cache from {path}: {e}")
        return None


def load_cache(path: str, max_age_seconds: int = None):
    """
    Charge le cache si le fichier existe et n'est pas expiré.

    Args:
        path (str): Chemin du fichier cache.
        max_age_seconds (int, optional): Durée maximale du cache en secondes.

    Returns:
        Any: Objet cache ou None si inexistant/expiré.
    """
    if is_cache_expired(path, max_age_seconds):
        return None
    return load_cache_file(path)


def save_cache(path: str, obj):
    """
    Sauvegarde un objet sérialisable dans le cache.

    Args:
        path (str): Chemin de destination.
        obj (Any): Objet à sauvegarder.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
            logger.info(f"[CACHE SAVE] Saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save cache to {path}: {e}")
