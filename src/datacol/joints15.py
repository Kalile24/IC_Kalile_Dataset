"""Mapping versionado dos 33 landmarks MediaPipe para as 15 juntas do modelo."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

JOINTS15_MAPPING_VERSION = "v1"

SKELETON_LIST = {
    "left_shoulder": 0,
    "right_shoulder": 1,
    "left_elbow": 2,
    "right_elbow": 3,
    "left_wrist": 4,
    "right_wrist": 5,
    "left_pinky": 6,
    "right_pinky": 7,
    "left_index": 8,
    "right_index": 9,
    "left_thumb": 10,
    "right_thumb": 11,
    "left_hip": 12,
    "right_hip": 13,
    "nose": 14,
}

MEDIAPIPE_INDICES = tuple(range(11, 25)) + (0,)


def extract_joints15(
    landmarks: Sequence[Sequence[float]],
) -> NDArray[np.float32]:
    """Converte landmarks MediaPipe para o contrato joints15 v1.

    Args:
        landmarks: Sequencia com 33 landmarks. Cada item deve fornecer ao menos
            as coordenadas ``x``, ``y`` e ``z`` nas tres primeiras posicoes.

    Returns:
        Array ``float32`` com shape ``(15, 3)``, ordenado conforme
        ``SKELETON_LIST`` e ``MEDIAPIPE_INDICES``.

    Raises:
        ValueError: Se a entrada nao tiver 33 landmarks ou tres coordenadas.
    """
    array = np.asarray(landmarks, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 33 or array.shape[1] < 3:
        raise ValueError(
            "landmarks must have shape (33, N), with N >= 3; "
            f"received {array.shape}"
        )
    return array[np.asarray(MEDIAPIPE_INDICES), :3].copy()
