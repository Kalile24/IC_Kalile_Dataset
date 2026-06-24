"""Criterios de aceite da captura e sincronizacao dos artefatos."""

import json
from datetime import datetime
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any, List

import cv2
from jsonschema import Draft202012Validator, FormatChecker
import numpy as np
import pytest

from datacol.capture_session import (
    build_session_metadata,
    capture_session,
    discover_cameras,
    suggest_session_id,
    validate_session,
)

FRAME_LEVELS = [20, 40, 60, 80, 100, 120]
FRAME_SIZE = (64, 48)


class FakeCapture:
    """Fonte deterministica compativel com a parte usada de VideoCapture."""

    def __init__(self, frame_levels: List[int]) -> None:
        width, height = FRAME_SIZE
        self._frames = [
            np.full((height, width, 3), level, dtype=np.uint8)
            for level in frame_levels
        ]
        self._index = 0
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def set(self, _property: int, _value: float) -> bool:
        return True

    def read(self) -> Any:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def release(self) -> None:
        self._open = False


class ClosedCapture:
    """Dispositivo que nao pode ser aberto."""

    def isOpened(self) -> bool:
        return False

    def read(self) -> Any:
        return False, None

    def release(self) -> None:
        return None


class OneFrameCapture(FakeCapture):
    """Entrega um quadro antes de simular desconexao."""

    def __init__(self) -> None:
        super().__init__([50])


class FakePose:
    """Pose sintetica cujo eixo x identifica o quadro de origem."""

    def process(self, rgb_frame: np.ndarray) -> Any:
        x_value = float(rgb_frame[0, 0, 0]) / 255.0
        landmarks = [
            SimpleNamespace(
                x=x_value,
                y=float(index) / 32.0,
                z=-x_value,
                visibility=1.0,
            )
            for index in range(33)
        ]
        return SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=landmarks)
        )

    def close(self) -> None:
        return None


@pytest.fixture(scope="module")
def captured_session(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("sessions")
    metadata = build_session_metadata(
        session_id="S01_20260613",
        participant="P01",
        script_id="R01",
        camera_model="synthetic-camera",
        resolution=FRAME_SIZE,
        fps_nominal=30,
        autofocus="disabled",
        camera_distance_m=2.2,
        camera_height="chest",
        zone_layout_version="v1",
        mediapipe_version="test",
        quaternion_world=[1.0, 0.0, 0.0, 0.0],
        date_iso="2026-06-13T12:00:00-03:00",
    )
    session_dir = capture_session(
        output_root=root,
        metadata=metadata,
        camera_index=0,
        preview=False,
        max_frames=len(FRAME_LEVELS),
        capture_factory=lambda _index: FakeCapture(FRAME_LEVELS),
        pose_factory=FakePose,
    )
    records = [
        json.loads(line)
        for line in (session_dir / "frames.jsonl").read_text().splitlines()
    ]
    with (session_dir / "skeleton.pkl").open("rb") as skeleton_file:
        skeleton = pickle.load(skeleton_file)
    return SimpleNamespace(
        path=session_dir,
        metadata=json.loads((session_dir / "meta.json").read_text()),
        records=records,
        skeleton=skeleton,
    )


def _video_frames(video_path: Any) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_artifact_frame_counts_are_equal(captured_session: Any) -> None:
    """PKL, JSONL e video devem conter exatamente o mesmo numero de quadros."""
    video_frames = _video_frames(captured_session.path / "video.mp4")
    expected = len(FRAME_LEVELS)
    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / "frames.schema.json"
    )
    frame_validator = Draft202012Validator(json.loads(schema_path.read_text()))

    assert len(captured_session.records) == expected
    assert captured_session.skeleton.shape == (expected, 15, 3)
    assert len(video_frames) == expected
    for record in captured_session.records:
        frame_validator.validate(record)


def test_frame_indices_and_timestamps_are_monotonic(
    captured_session: Any,
) -> None:
    """frame_idx deve ser contiguo e t_mono estritamente crescente."""
    frame_indices = [record["frame_idx"] for record in captured_session.records]
    timestamps = [record["t_mono"] for record in captured_session.records]

    assert frame_indices == list(range(len(FRAME_LEVELS)))
    assert timestamps[0] == 0.0
    assert all(
        current < following
        for current, following in zip(timestamps, timestamps[1:])
    )


def test_effective_fps_is_recorded_after_capture(
    captured_session: Any,
) -> None:
    """meta.json final deve registrar fps_effective positivo e nao nulo."""
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "meta.schema.json"
    schema = json.loads(schema_path.read_text())
    Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    ).validate(captured_session.metadata)

    assert captured_session.metadata["camera"]["fps_effective"] > 0


def test_video_and_skeleton_replay_stay_synchronized(
    captured_session: Any,
) -> None:
    """A sobreposicao deve permanecer no quadro correspondente em toda sessao."""
    video_frames = _video_frames(captured_session.path / "video.mp4")
    video_levels = np.asarray([frame.mean() for frame in video_frames])
    json_pose_x = np.asarray(
        [record["joints15"][0][0] for record in captured_session.records]
    )
    pkl_pose_x = captured_session.skeleton[:, 0, 0]

    assert np.all(np.diff(video_levels) > 0)
    assert np.allclose(json_pose_x, np.asarray(FRAME_LEVELS) / 255.0)
    assert np.allclose(pkl_pose_x, json_pose_x)
    assert np.corrcoef(video_levels, json_pose_x)[0, 1] > 0.99


def test_validate_session_accepts_complete_capture(captured_session: Any) -> None:
    """O validador deve confirmar schemas, shapes e cardinalidades."""
    report = validate_session(captured_session.path)

    assert report["valid"] is True
    assert report["frame_count"] == len(FRAME_LEVELS)
    assert report["errors"] == []


def test_suggest_session_id_uses_next_global_number(tmp_path: Path) -> None:
    """A sugestao deve evitar IDs existentes, inclusive tentativas incompletas."""
    (tmp_path / "S01_20260610").mkdir()
    (tmp_path / "S06_20260612").mkdir()
    (tmp_path / "notes").mkdir()

    suggested = suggest_session_id(
        tmp_path,
        date=datetime(2026, 6, 13),
    )

    assert suggested == "S07_20260613"


def test_discover_cameras_returns_only_readable_indices() -> None:
    """A descoberta deve exigir abertura e leitura de ao menos um quadro."""
    def capture_factory(index: int) -> Any:
        if index == 2:
            return FakeCapture([50])
        return ClosedCapture()

    assert discover_cameras(
        max_index=4,
        capture_factory=capture_factory,
    ) == [{"index": 2, "width": 64, "height": 48}]


def test_failed_capture_removes_only_empty_session(tmp_path: Path) -> None:
    """Falha antes do primeiro quadro nao deve deixar diretorio residual."""
    metadata = build_session_metadata(
        session_id="S01_20260613",
        participant="P01",
        script_id="R01",
        camera_model="closed-camera",
        resolution=FRAME_SIZE,
        fps_nominal=30,
        autofocus="disabled",
        camera_distance_m=2.2,
        camera_height="chest",
        zone_layout_version="v1",
        mediapipe_version="test",
        quaternion_world=[1.0, 0.0, 0.0, 0.0],
        date_iso="2026-06-13T12:00:00-03:00",
    )

    with pytest.raises(RuntimeError, match="Could not open camera"):
        capture_session(
            output_root=tmp_path,
            metadata=metadata,
            camera_index=0,
            preview=False,
            capture_factory=lambda _index: ClosedCapture(),
            pose_factory=FakePose,
        )

    assert not (tmp_path / metadata["session_id"]).exists()


def test_failed_capture_preserves_session_with_frames(tmp_path: Path) -> None:
    """Uma tentativa com ao menos um quadro deve permanecer para auditoria."""
    metadata = build_session_metadata(
        session_id="S02_20260613",
        participant="P01",
        script_id="R01",
        camera_model="disconnecting-camera",
        resolution=FRAME_SIZE,
        fps_nominal=30,
        autofocus="disabled",
        camera_distance_m=2.2,
        camera_height="chest",
        zone_layout_version="v1",
        mediapipe_version="test",
        quaternion_world=[1.0, 0.0, 0.0, 0.0],
        date_iso="2026-06-13T12:00:00-03:00",
    )

    with pytest.raises(RuntimeError, match="At least two frames"):
        capture_session(
            output_root=tmp_path,
            metadata=metadata,
            camera_index=0,
            preview=False,
            capture_factory=lambda _index: OneFrameCapture(),
            pose_factory=FakePose,
        )

    session_dir = tmp_path / metadata["session_id"]
    assert session_dir.is_dir()
    assert (session_dir / "frames.jsonl").is_file()
    assert (session_dir / "video.mp4").is_file()
