"""Logger sincronizado de webcam e MediaPipe Pose."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datacol.joints15 import JOINTS15_MAPPING_VERSION, extract_joints15

DEFAULT_QUATERNION_WORLD = (
    0.14070565,
    -0.15007018,
    -0.7552408,
    0.62232804,
)
MISSING_LANDMARKS = np.zeros((33, 4), dtype=np.float32)
MISSING_JOINTS15 = np.zeros((15, 3), dtype=np.float32)
SESSION_FILES = ("meta.json", "frames.jsonl", "skeleton.pkl", "video.mp4")
CAMERA_SCAN_LIMIT = 10


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _validate_metadata(metadata: Dict[str, Any]) -> None:
    required = {
        "session_id",
        "participant",
        "script_id",
        "date_iso",
        "camera",
        "geometry",
        "pipeline",
        "files",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"metadata is missing fields: {sorted(missing)}")
    extra = set(metadata).difference(required)
    if extra:
        raise ValueError(f"metadata has unsupported fields: {sorted(extra)}")

    patterns = {
        "session_id": r"S[0-9]+_[0-9]{8}",
        "participant": r"P[0-9]+",
        "script_id": r"R[0-9]+",
    }
    examples = {
        "session_id": "S01_20260613",
        "participant": "P01",
        "script_id": "R01",
    }
    for field, pattern in patterns.items():
        if not isinstance(metadata[field], str) or re.fullmatch(
            pattern, metadata[field]
        ) is None:
            raise ValueError(
                f"invalid {field}={metadata[field]!r}; expected format "
                f"{pattern!r}, for example {examples[field]!r}"
            )

    try:
        date_value = datetime.fromisoformat(metadata["date_iso"])
    except (TypeError, ValueError) as exc:
        raise ValueError("date_iso must be a valid ISO 8601 datetime") from exc
    if date_value.utcoffset() is None:
        raise ValueError("date_iso must include a UTC offset")

    camera = metadata["camera"]
    camera_fields = {
        "model",
        "resolution",
        "fps_nominal",
        "fps_effective",
        "autofocus",
    }
    if set(camera) != camera_fields:
        raise ValueError(f"camera fields must be exactly {sorted(camera_fields)}")
    if not isinstance(camera["model"], str) or not camera["model"]:
        raise ValueError("camera.model must be a non-empty string")
    if not isinstance(camera["autofocus"], str) or not camera["autofocus"]:
        raise ValueError("camera.autofocus must be a non-empty string")

    resolution = camera.get("resolution")
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or any(not isinstance(value, int) or value <= 0 for value in resolution)
    ):
        raise ValueError("camera.resolution must be [width, height] with positive integers")
    fps_nominal = float(camera.get("fps_nominal", 0))
    if not math.isfinite(fps_nominal) or fps_nominal <= 0:
        raise ValueError("camera.fps_nominal must be greater than zero")
    fps_effective = camera["fps_effective"]
    if fps_effective is not None:
        effective_value = float(fps_effective)
        if not math.isfinite(effective_value) or effective_value <= 0:
            raise ValueError("camera.fps_effective must be null or greater than zero")

    geometry = metadata["geometry"]
    geometry_fields = {
        "camera_distance_m",
        "camera_height",
        "zone_layout_version",
    }
    if set(geometry) != geometry_fields:
        raise ValueError(f"geometry fields must be exactly {sorted(geometry_fields)}")
    camera_distance = float(geometry["camera_distance_m"])
    if not math.isfinite(camera_distance) or camera_distance <= 0:
        raise ValueError("geometry.camera_distance_m must be greater than zero")
    for field in ("camera_height", "zone_layout_version"):
        if not isinstance(geometry[field], str) or not geometry[field]:
            raise ValueError(f"geometry.{field} must be a non-empty string")

    pipeline = metadata["pipeline"]
    pipeline_fields = {
        "mediapipe_version",
        "joints15_mapping_version",
        "quaternion_world",
    }
    if set(pipeline) != pipeline_fields:
        raise ValueError(f"pipeline fields must be exactly {sorted(pipeline_fields)}")
    if not isinstance(pipeline["mediapipe_version"], str) or not pipeline[
        "mediapipe_version"
    ]:
        raise ValueError("pipeline.mediapipe_version must be a non-empty string")

    quaternion = pipeline.get("quaternion_world")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError("pipeline.quaternion_world must be [w, x, y, z]")
    if any(not math.isfinite(float(value)) for value in quaternion):
        raise ValueError("pipeline.quaternion_world must contain finite numbers")
    if pipeline.get("joints15_mapping_version") != JOINTS15_MAPPING_VERSION:
        raise ValueError(
            "pipeline.joints15_mapping_version must match "
            f"{JOINTS15_MAPPING_VERSION!r}"
        )

    expected_files = {
        "frames": "frames.jsonl",
        "skeleton": "skeleton.pkl",
        "video": "video.mp4",
    }
    if metadata["files"] != expected_files:
        raise ValueError(f"files must be exactly {expected_files}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def suggest_session_id(
    output_root: Path,
    date: Optional[datetime] = None,
) -> str:
    """Sugere o proximo identificador de sessao sem sobrescrever diretorios.

    Args:
        output_root: Diretorio que contem as sessoes.
        date: Data usada no sufixo; por padrao, a data local atual.

    Returns:
        Identificador ``SNN_YYYYMMDD`` com numero superior ao maior ja usado.
    """
    output_root = Path(output_root)
    date_suffix = (date or datetime.now().astimezone()).strftime("%Y%m%d")
    highest_number = 0
    if output_root.exists():
        for path in output_root.iterdir():
            match = re.fullmatch(r"S([0-9]+)_([0-9]{8})", path.name)
            if path.is_dir() and match:
                highest_number = max(highest_number, int(match.group(1)))
    return f"S{highest_number + 1:02d}_{date_suffix}"


def discover_cameras(
    max_index: int = CAMERA_SCAN_LIMIT,
    capture_factory: Optional[Callable[[int], Any]] = None,
) -> List[Dict[str, Any]]:
    """Sonda indices OpenCV e retorna apenas cameras que entregam um quadro.

    Args:
        max_index: Testa indices no intervalo ``range(max_index)``.
        capture_factory: Fabrica opcional compativel com ``cv2.VideoCapture``.

    Returns:
        Lista de mapas com ``index``, ``width`` e ``height``.
    """
    if max_index <= 0:
        raise ValueError("max_index must be greater than zero")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the project dependencies first."
        ) from exc

    capture_builder = capture_factory or cv2.VideoCapture
    cameras = []
    for index in range(max_index):
        capture = capture_builder(index)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None or frame.ndim < 2:
                continue
            height, width = frame.shape[:2]
            cameras.append({"index": index, "width": width, "height": height})
        finally:
            capture.release()
    return cameras


def _draw_capture_hud(
    frame: np.ndarray,
    frame_count: int,
    elapsed_seconds: float,
    pose_detected: bool,
) -> np.ndarray:
    import cv2

    display = frame.copy()
    current_fps = frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
    status = "POSE OK" if pose_detected else "POSE MISSING"
    status_color = (0, 220, 0) if pose_detected else (0, 0, 255)
    lines = (
        f"REC  frame {frame_count}",
        f"time {elapsed_seconds:7.1f}s  fps {current_fps:5.1f}",
        status,
        "q / Esc: stop",
    )
    y = 30
    for line_index, line in enumerate(lines):
        color = status_color if line_index == 2 else (255, 255, 255)
        cv2.putText(
            display,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30
    return display


def _remove_empty_session(session_dir: Path) -> None:
    for filename in SESSION_FILES:
        path = session_dir / filename
        if path.exists():
            path.unlink()
    try:
        session_dir.rmdir()
    except OSError:
        pass


def _create_pose_estimator() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Install the project dependencies first."
        ) from exc

    try:
        pose_class = mp.solutions.pose.Pose
    except AttributeError as exc:
        raise RuntimeError(
            "This logger requires the MediaPipe Solutions Pose API."
        ) from exc

    return pose_class(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _extract_pose_arrays(result: Any) -> Tuple[np.ndarray, np.ndarray]:
    if result is None or result.pose_landmarks is None:
        return MISSING_LANDMARKS.copy(), MISSING_JOINTS15.copy()

    landmarks = np.asarray(
        [
            [landmark.x, landmark.y, landmark.z, landmark.visibility]
            for landmark in result.pose_landmarks.landmark
        ],
        dtype=np.float32,
    )
    if landmarks.shape != (33, 4):
        raise ValueError(
            "MediaPipe Pose must return exactly 33 landmarks; "
            f"received {landmarks.shape}"
        )
    return landmarks, extract_joints15(landmarks)


def _effective_fps(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        raise RuntimeError("At least two frames are required to compute fps_effective")
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        raise RuntimeError("Capture timestamps must be strictly increasing")
    return (len(timestamps) - 1) / elapsed


def validate_session(session_dir: Path) -> Dict[str, Any]:
    """Valida schemas, indices, shapes e cardinalidade dos quatro artefatos.

    Args:
        session_dir: Diretorio de uma sessao finalizada.

    Returns:
        Relatorio com ``valid``, contagens, FPS e lista ``errors``.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the project dependencies first."
        ) from exc

    session_dir = Path(session_dir)
    errors = []
    paths = {name: session_dir / name for name in SESSION_FILES}
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing file: {name}")
    if errors:
        return {
            "session_dir": str(session_dir),
            "valid": False,
            "errors": errors,
        }

    try:
        metadata = json.loads(paths["meta.json"].read_text(encoding="utf-8"))
        _validate_metadata(metadata)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        metadata = {}
        errors.append(f"invalid meta.json: {exc}")

    records = []
    try:
        with paths["frames.jsonl"].open(encoding="utf-8") as frames_file:
            for line_number, line in enumerate(frames_file, start=1):
                record = json.loads(line)
                landmarks = np.asarray(record.get("landmarks_raw"))
                joints = np.asarray(record.get("joints15"))
                if landmarks.shape != (33, 4):
                    errors.append(
                        f"frames.jsonl line {line_number}: "
                        f"landmarks_raw shape is {landmarks.shape}"
                    )
                if joints.shape != (15, 3):
                    errors.append(
                        f"frames.jsonl line {line_number}: "
                        f"joints15 shape is {joints.shape}"
                    )
                records.append(record)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid frames.jsonl: {exc}")

    frame_indices = [record.get("frame_idx") for record in records]
    if frame_indices != list(range(len(records))):
        errors.append("frame_idx is not contiguous and zero-based")
    timestamps = [record.get("t_mono") for record in records]
    if timestamps:
        if timestamps[0] != 0:
            errors.append("first t_mono is not zero")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in timestamps
        ):
            errors.append("t_mono contains non-finite or non-numeric values")
        elif any(
            current >= following
            for current, following in zip(timestamps, timestamps[1:])
        ):
            errors.append("t_mono is not strictly increasing")
    else:
        errors.append("frames.jsonl contains no frames")

    skeleton = None
    try:
        with paths["skeleton.pkl"].open("rb") as skeleton_file:
            skeleton = np.asarray(pickle.load(skeleton_file))
        if skeleton.ndim != 3 or skeleton.shape[1:] != (15, 3):
            errors.append(f"skeleton.pkl shape is {skeleton.shape}, expected (N, 15, 3)")
    except (OSError, ValueError, TypeError, pickle.UnpicklingError) as exc:
        errors.append(f"invalid skeleton.pkl: {exc}")

    video = cv2.VideoCapture(str(paths["video.mp4"]))
    video_count = 0
    if not video.isOpened():
        errors.append("video.mp4 could not be opened")
    else:
        while True:
            ok, _frame = video.read()
            if not ok:
                break
            video_count += 1
    video.release()

    jsonl_count = len(records)
    skeleton_count = int(skeleton.shape[0]) if skeleton is not None and skeleton.ndim else 0
    if len({jsonl_count, skeleton_count, video_count}) != 1:
        errors.append(
            "artifact frame counts differ: "
            f"jsonl={jsonl_count}, pkl={skeleton_count}, video={video_count}"
        )
    if skeleton is not None and skeleton.shape == (jsonl_count, 15, 3):
        json_joints = np.asarray(
            [record.get("joints15") for record in records],
            dtype=np.float32,
        )
        if json_joints.shape != skeleton.shape or not np.allclose(
            json_joints, skeleton, equal_nan=False
        ):
            errors.append("skeleton.pkl differs from frames.jsonl joints15")

    fps_effective = metadata.get("camera", {}).get("fps_effective")
    if fps_effective is None:
        errors.append("meta.json camera.fps_effective is null")

    return {
        "session_dir": str(session_dir),
        "session_id": metadata.get("session_id"),
        "valid": not errors,
        "frame_count": jsonl_count,
        "jsonl_frames": jsonl_count,
        "skeleton_frames": skeleton_count,
        "video_frames": video_count,
        "fps_effective": fps_effective,
        "errors": errors,
    }


def build_session_metadata(
    session_id: str,
    participant: str,
    script_id: str,
    camera_model: str,
    resolution: Tuple[int, int],
    fps_nominal: float,
    autofocus: str,
    camera_distance_m: float,
    camera_height: str,
    zone_layout_version: str,
    mediapipe_version: str,
    quaternion_world: Sequence[float],
    date_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Monta o documento validavel por ``schemas/meta.schema.json``.

    Args:
        session_id: Identificador no formato ``SNN_YYYYMMDD``.
        participant: Identificador do participante, por exemplo ``P01``.
        script_id: Identificador do roteiro, por exemplo ``R02``.
        camera_model: Nome registrado do dispositivo de captura.
        resolution: Par ``(largura, altura)`` em pixels.
        fps_nominal: Taxa nominal solicitada ao dispositivo.
        autofocus: Estado ou metodo usado para controlar o foco.
        camera_distance_m: Distancia camera-participante em metros.
        camera_height: Descricao da altura da camera.
        zone_layout_version: Versao do layout das zonas de pega.
        mediapipe_version: Versao instalada do MediaPipe.
        quaternion_world: Quaternion ``[w, x, y, z]`` de camera para mundo.
        date_iso: Data ISO 8601 com fuso; quando ausente, usar o inicio atual.

    Returns:
        Dicionario completo de metadados, com ``fps_effective`` inicialmente
        nulo e os nomes canonicos dos tres artefatos de captura.
    """
    if len(resolution) != 2 or any(int(value) <= 0 for value in resolution):
        raise ValueError("resolution must contain positive width and height")
    if fps_nominal <= 0:
        raise ValueError("fps_nominal must be greater than zero")
    if camera_distance_m <= 0:
        raise ValueError("camera_distance_m must be greater than zero")
    if len(quaternion_world) != 4:
        raise ValueError("quaternion_world must contain [w, x, y, z]")

    metadata = {
        "session_id": session_id,
        "participant": participant,
        "script_id": script_id,
        "date_iso": date_iso or datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera": {
            "model": camera_model,
            "resolution": [int(resolution[0]), int(resolution[1])],
            "fps_nominal": float(fps_nominal),
            "fps_effective": None,
            "autofocus": autofocus,
        },
        "geometry": {
            "camera_distance_m": float(camera_distance_m),
            "camera_height": camera_height,
            "zone_layout_version": zone_layout_version,
        },
        "pipeline": {
            "mediapipe_version": mediapipe_version,
            "joints15_mapping_version": JOINTS15_MAPPING_VERSION,
            "quaternion_world": [float(value) for value in quaternion_world],
        },
        "files": {
            "frames": "frames.jsonl",
            "skeleton": "skeleton.pkl",
            "video": "video.mp4",
        },
    }
    _validate_metadata(metadata)
    return metadata


def capture_session(
    output_root: Path,
    metadata: Dict[str, Any],
    camera_index: Optional[int] = None,
    *,
    preview: bool = True,
    max_frames: Optional[int] = None,
    capture_factory: Optional[Callable[[int], Any]] = None,
    pose_factory: Optional[Callable[[], Any]] = None,
) -> Path:
    """Captura uma sessao e grava quatro artefatos sincronizados.

    Args:
        output_root: Diretorio ``sessions`` sob o qual criar ``session_id``.
        metadata: Documento produzido por ``build_session_metadata``.
        camera_index: Indice OpenCV da webcam. Quando ``None``, usa a primeira
            camera que ``discover_cameras`` conseguir abrir e ler.
        preview: Exibe o video ao vivo; ``q`` ou ``Esc`` encerra a captura.
        max_frames: Limite opcional, util para pilotos e testes automatizados.
        capture_factory: Fabrica opcional compativel com ``cv2.VideoCapture``.
        pose_factory: Fabrica opcional de um estimador com metodo ``process``.

    Returns:
        Caminho do diretorio criado para a sessao.

    Side Effects:
        Grava ``meta.json``, ``frames.jsonl``, ``skeleton.pkl`` e
        ``video.mp4``. Cada quadro usa o mesmo ``frame_idx`` zero-based nos
        tres fluxos. ``skeleton.pkl`` contem um array ``(N, 15, 3)``.

    Raises:
        FileExistsError: Se o diretorio da sessao ja existir.
        RuntimeError: Se camera, encoder, MediaPipe ou timestamps falharem.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the project dependencies first."
        ) from exc

    _validate_metadata(metadata)
    if max_frames is not None and max_frames < 2:
        raise ValueError("max_frames must be at least 2")

    output_root = Path(output_root)
    session_dir = output_root / metadata["session_id"]
    if session_dir.exists():
        raise FileExistsError(f"session directory already exists: {session_dir}")
    session_dir.mkdir(parents=True)

    session_metadata = deepcopy(metadata)
    meta_path = session_dir / "meta.json"
    frames_path = session_dir / session_metadata["files"]["frames"]
    skeleton_path = session_dir / session_metadata["files"]["skeleton"]
    video_path = session_dir / session_metadata["files"]["video"]
    capture_builder = capture_factory or cv2.VideoCapture
    pose_builder = pose_factory or _create_pose_estimator
    if camera_index is None:
        cameras = discover_cameras(capture_factory=capture_factory)
        if not cameras:
            _remove_empty_session(session_dir)
            raise RuntimeError("No readable camera was found")
        camera_index = cameras[0]["index"]

    capture = None
    pose = None
    writer = None
    timestamps = []
    skeleton_frames = []
    first_timestamp_ns = None
    frame_idx = 0
    video_size = None

    width, height = session_metadata["camera"]["resolution"]
    fps_nominal = session_metadata["camera"]["fps_nominal"]

    try:
        _write_json(meta_path, session_metadata)
        capture = capture_builder(camera_index)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera index {camera_index}")

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps_nominal)
            pose = pose_builder()

            with frames_path.open("w", encoding="utf-8") as frames_file:
                while True:
                    ok, frame = capture.read()
                    captured_ns = time.perf_counter_ns()
                    if not ok or frame is None:
                        break

                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise RuntimeError(
                            "Camera frame must have shape (H, W, 3); "
                            f"received {frame.shape}"
                        )

                    actual_height, actual_width = frame.shape[:2]
                    if writer is None:
                        video_size = (actual_width, actual_height)
                        session_metadata["camera"]["resolution"] = [
                            actual_width,
                            actual_height,
                        ]
                        writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps_nominal,
                            (actual_width, actual_height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Could not create video file: {video_path}"
                            )

                    if (actual_width, actual_height) != video_size:
                        raise RuntimeError("Camera resolution changed during capture")

                    if first_timestamp_ns is None:
                        first_timestamp_ns = captured_ns
                    t_mono = (captured_ns - first_timestamp_ns) / 1_000_000_000
                    if timestamps and t_mono <= timestamps[-1]:
                        t_mono = float(np.nextafter(timestamps[-1], np.inf))

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb_frame)
                    pose_detected = (
                        result is not None and result.pose_landmarks is not None
                    )
                    landmarks_raw, joints15 = _extract_pose_arrays(result)

                    record = {
                        "frame_idx": frame_idx,
                        "t_mono": t_mono,
                        "landmarks_raw": landmarks_raw.tolist(),
                        "joints15": joints15.tolist(),
                    }
                    record_line = json.dumps(
                        record,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    writer.write(frame)
                    frames_file.write(record_line + "\n")
                    timestamps.append(t_mono)
                    skeleton_frames.append(joints15)
                    frame_idx += 1

                    if preview:
                        display = _draw_capture_hud(
                            frame,
                            frame_count=frame_idx,
                            elapsed_seconds=t_mono,
                            pose_detected=pose_detected,
                        )
                        cv2.imshow(
                            "HRC data collection - q/Esc to stop",
                            display,
                        )
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    if max_frames is not None and frame_idx >= max_frames:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            if pose is not None and hasattr(pose, "close"):
                pose.close()
            if preview:
                cv2.destroyAllWindows()

        session_metadata["camera"]["fps_effective"] = _effective_fps(timestamps)
        skeleton = np.stack(skeleton_frames).astype(np.float32, copy=False)
        with skeleton_path.open("wb") as skeleton_file:
            pickle.dump(skeleton, skeleton_file, protocol=pickle.HIGHEST_PROTOCOL)
        _write_json(meta_path, session_metadata)
        return session_dir
    except BaseException:
        if frame_idx == 0:
            _remove_empty_session(session_dir)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Executa a interface de linha de comando do logger.

    Args:
        argv: Argumentos sem o nome do programa, ou ``None`` para ``sys.argv``.

    Returns:
        Codigo de saida do processo.
    """
    parser = argparse.ArgumentParser(
        description="Capture a synchronized webcam and MediaPipe Pose session."
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        help="optional SNN_YYYYMMDD; the next ID is suggested when omitted",
    )
    parser.add_argument("--participant")
    parser.add_argument("--script-id")
    parser.add_argument("--camera-model")
    parser.add_argument(
        "--camera-index",
        type=int,
        help="OpenCV camera index; auto-detected when omitted",
    )
    parser.add_argument(
        "--camera-scan-limit",
        type=_positive_int,
        default=CAMERA_SCAN_LIMIT,
    )
    parser.add_argument("--width", type=_positive_int, default=1920)
    parser.add_argument("--height", type=_positive_int, default=1080)
    parser.add_argument("--fps", type=_positive_float, default=30.0)
    parser.add_argument("--autofocus", default="locked_v4l2")
    parser.add_argument("--camera-distance-m", type=_positive_float)
    parser.add_argument("--camera-height", default="chest")
    parser.add_argument("--zone-layout-version", default="v1")
    parser.add_argument("--output-root", type=Path, default=Path("sessions"))
    parser.add_argument("--date-iso")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="list readable OpenCV camera indices and exit",
    )
    parser.add_argument(
        "--suggest-session-id",
        action="store_true",
        help="print the next available session ID and exit",
    )
    parser.add_argument(
        "--validate-session",
        type=Path,
        help="validate an existing session directory and exit",
    )
    parser.add_argument(
        "--quaternion-world",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        default=DEFAULT_QUATERNION_WORLD,
    )
    args = parser.parse_args(argv)

    if args.list_cameras:
        cameras = discover_cameras(max_index=args.camera_scan_limit)
        if not cameras:
            print("No readable cameras found.")
            return 1
        for camera in cameras:
            print(
                f"index={camera['index']} "
                f"resolution={camera['width']}x{camera['height']}"
            )
        return 0

    if args.suggest_session_id:
        print(suggest_session_id(args.output_root))
        return 0

    if args.validate_session is not None:
        report = validate_session(args.validate_session)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0 if report["valid"] else 1

    missing_arguments = [
        option
        for option, value in (
            ("--participant", args.participant),
            ("--script-id", args.script_id),
            ("--camera-model", args.camera_model),
            ("--camera-distance-m", args.camera_distance_m),
        )
        if value is None
    ]
    if missing_arguments:
        parser.error(
            "capture mode requires: " + ", ".join(missing_arguments)
        )

    try:
        import mediapipe as mp
    except ImportError:
        parser.error("mediapipe is not installed")

    session_id = args.session_id or suggest_session_id(args.output_root)
    if args.session_id is None:
        print(f"Using suggested session ID: {session_id}")

    camera_index = args.camera_index
    if camera_index is None:
        cameras = discover_cameras(max_index=args.camera_scan_limit)
        if not cameras:
            parser.error("no readable camera was found")
        camera_index = cameras[0]["index"]
        print(
            f"Using camera index {camera_index} "
            f"({cameras[0]['width']}x{cameras[0]['height']})"
        )

    metadata = build_session_metadata(
        session_id=session_id,
        participant=args.participant,
        script_id=args.script_id,
        camera_model=args.camera_model,
        resolution=(args.width, args.height),
        fps_nominal=args.fps,
        autofocus=args.autofocus,
        camera_distance_m=args.camera_distance_m,
        camera_height=args.camera_height,
        zone_layout_version=args.zone_layout_version,
        mediapipe_version=mp.__version__,
        quaternion_world=args.quaternion_world,
        date_iso=args.date_iso,
    )
    session_dir = capture_session(
        output_root=args.output_root,
        metadata=metadata,
        camera_index=camera_index,
        preview=not args.no_preview,
        max_frames=args.max_frames,
    )
    print(session_dir)
    report = validate_session(session_dir)
    if report["valid"]:
        print(
            "Validation OK: "
            f"{report['frame_count']} frames, "
            f"{report['fps_effective']:.2f} effective FPS"
        )
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
