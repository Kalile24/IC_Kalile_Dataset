"""Contratos publicos do repositorio hrc-data-collection."""

INTENTION_LIST = {
    "no_action": 0,
    "get_connectors": 1,
    "get_screws": 2,
    "get_wheels": 3,
}

ANNOTATION_LABELS = tuple(INTENTION_LIST) + ("ignore",)

__all__ = ["ANNOTATION_LABELS", "INTENTION_LIST"]
