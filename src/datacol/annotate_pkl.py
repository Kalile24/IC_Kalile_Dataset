"""Anotador quadro a quadro com replay sincronizado e overlay do esqueleto."""

import argparse
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datacol import ANNOTATION_LABELS

IntervalMap = Dict[str, Dict[str, List[int]]]
PlanEvent = Dict[str, Any]
PlanEventsDocument = Dict[str, List[PlanEvent]]
PLAN_EVENT_BEGIN_FOUR_TUBES = "begin_four_tubes"

DEFAULT_KEY_BINDINGS = {
    "no_action": "0",
    "get_connectors": "1",
    "get_screws": "2",
    "get_wheels": "3",
    "ignore": "i",
}

TORSO_CONNECTIONS = (
    (0, 1),
    (1, 13),
    (13, 12),
    (12, 0),
)

ARM_CONNECTIONS = (
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
)

LABEL_COLORS = {
    "no_action": (160, 160, 160),
    "get_connectors": (0, 200, 255),
    "get_screws": (255, 120, 0),
    "get_wheels": (180, 0, 255),
    "ignore": (0, 0, 255),
}

PANEL_BACKGROUND = (24, 24, 28)
PANEL_BORDER = (70, 70, 78)
BUTTON_BACKGROUND = (52, 52, 60)
BUTTON_HOVER = (72, 72, 82)
TEXT_PRIMARY = (240, 240, 240)
TEXT_SECONDARY = (170, 170, 180)

Rect = Tuple[int, int, int, int]


def empty_annotations() -> IntervalMap:
    """Cria o documento vazio com todas as chaves do schema."""
    return {
        label: {"start": [], "end": []}
        for label in ANNOTATION_LABELS
    }


def labels_to_annotations(labels: Sequence[str]) -> IntervalMap:
    """Comprime um rotulo por quadro em intervalos inclusivos.

    Args:
        labels: Rotulo de cada quadro, na ordem de ``frame_idx``.

    Returns:
        Mapa no formato de ``annotations.schema.json``.
    """
    annotations = empty_annotations()
    if not labels:
        return annotations

    invalid = set(labels).difference(ANNOTATION_LABELS)
    if invalid:
        raise ValueError(f"unknown annotation labels: {sorted(invalid)}")

    start = 0
    current = labels[0]
    for frame_idx in range(1, len(labels) + 1):
        if frame_idx == len(labels) or labels[frame_idx] != current:
            annotations[current]["start"].append(start)
            annotations[current]["end"].append(frame_idx - 1)
            if frame_idx < len(labels):
                start = frame_idx
                current = labels[frame_idx]
    return annotations


def annotations_to_labels(
    annotations: IntervalMap,
    frame_count: int,
) -> List[str]:
    """Expande intervalos validados para um rotulo por quadro."""
    validate_annotations(annotations, frame_count)
    labels: List[Optional[str]] = [None] * frame_count
    for label in ANNOTATION_LABELS:
        starts = annotations[label]["start"]
        ends = annotations[label]["end"]
        for start, end in zip(starts, ends):
            labels[start : end + 1] = [label] * (end - start + 1)
    return [label for label in labels if label is not None]


def validate_annotations(
    annotations: IntervalMap,
    frame_count: int,
) -> None:
    """Valida cobertura e consistencia dos intervalos de uma sessao.

    Args:
        annotations: Mapa com as chaves de ``ANNOTATION_LABELS`` e listas
            paralelas ``start``/``end``. Os limites sao zero-based e inclusivos.
        frame_count: Numero total de quadros da sessao.

    Returns:
        ``None`` quando todos os quadros pertencem a exatamente um intervalo,
        sem lacunas ou sobreposicoes.

    Raises:
        ValueError: Para rotulos invalidos, limites fora da sessao, listas de
            tamanhos diferentes, lacunas ou sobreposicoes.
    """
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if not isinstance(annotations, dict):
        raise ValueError("annotations must be an object")
    if set(annotations) != set(ANNOTATION_LABELS):
        missing = set(ANNOTATION_LABELS).difference(annotations)
        extra = set(annotations).difference(ANNOTATION_LABELS)
        raise ValueError(
            f"annotation labels mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    owner: List[Optional[str]] = [None] * frame_count
    for label in ANNOTATION_LABELS:
        intervals = annotations[label]
        if not isinstance(intervals, dict) or set(intervals) != {"start", "end"}:
            raise ValueError(f"{label} must contain only start and end")
        starts = intervals["start"]
        ends = intervals["end"]
        if not isinstance(starts, list) or not isinstance(ends, list):
            raise ValueError(f"{label} start and end must be lists")
        if len(starts) != len(ends):
            raise ValueError(f"{label} start and end lengths differ")

        previous_end = -1
        for interval_index, (start, end) in enumerate(zip(starts, ends)):
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                raise ValueError(f"{label} interval indices must be integers")
            if start < 0 or end < start or end >= frame_count:
                raise ValueError(
                    f"{label} interval {interval_index} is outside frame range"
                )
            if start <= previous_end:
                raise ValueError(f"{label} intervals are not sorted and disjoint")
            previous_end = end
            for frame_idx in range(start, end + 1):
                if owner[frame_idx] is not None:
                    raise ValueError(
                        f"frame {frame_idx} overlaps {owner[frame_idx]} and {label}"
                    )
                owner[frame_idx] = label

    uncovered = [index for index, label in enumerate(owner) if label is None]
    if uncovered:
        preview = uncovered[:10]
        raise ValueError(f"frames are not fully covered; first gaps: {preview}")


def assign_interval(
    labels: List[str],
    start: int,
    end: int,
    label: str,
) -> Tuple[int, int]:
    """Atribui um rotulo a um intervalo inclusivo, aceitando selecao reversa."""
    if label not in ANNOTATION_LABELS:
        raise ValueError(f"unknown annotation label: {label}")
    if not labels:
        raise ValueError("labels cannot be empty")
    interval_start, interval_end = sorted((start, end))
    if interval_start < 0 or interval_end >= len(labels):
        raise ValueError("interval is outside frame range")
    labels[interval_start : interval_end + 1] = [label] * (
        interval_end - interval_start + 1
    )
    return interval_start, interval_end


def save_annotations(path: Path, annotations: IntervalMap, frame_count: int) -> None:
    """Valida e grava ``annotations.json`` de forma deterministica."""
    validate_annotations(annotations, frame_count)
    Path(path).write_text(
        json.dumps(annotations, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_annotations(path: Path, frame_count: int) -> IntervalMap:
    """Carrega e valida um arquivo de anotacoes."""
    annotations = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_annotations(annotations, frame_count)
    return annotations


def empty_plan_events() -> PlanEventsDocument:
    """Cria o documento vazio de eventos externos do PlanGraph."""
    return {"events": []}


def validate_plan_events(
    document: PlanEventsDocument,
    frame_count: int,
    labels: Optional[Sequence[str]] = None,
) -> None:
    """Valida eventos externos e sua posição na linha do tempo.

    ``begin_four_tubes`` deve ser único e marcado no primeiro quadro do bloco
    ``ignore`` que representa a entrega dos quatro tubos curtos.
    """
    if not isinstance(document, dict) or set(document) != {"events"}:
        raise ValueError("plan events must contain only the events list")
    events = document["events"]
    if not isinstance(events, list):
        raise ValueError("plan events must be a list")
    if labels is not None and len(labels) != frame_count:
        raise ValueError("labels length must match frame_count")

    previous_frame = -1
    seen = set()
    for event in events:
        if not isinstance(event, dict) or set(event) != {"frame_idx", "event"}:
            raise ValueError("each plan event must contain frame_idx and event")
        frame_idx = event["frame_idx"]
        event_name = event["event"]
        if (
            not isinstance(frame_idx, int)
            or isinstance(frame_idx, bool)
            or frame_idx < 0
            or frame_idx >= frame_count
        ):
            raise ValueError("plan event frame_idx is outside frame range")
        if event_name != PLAN_EVENT_BEGIN_FOUR_TUBES:
            raise ValueError(f"unknown plan event: {event_name!r}")
        if frame_idx <= previous_frame:
            raise ValueError("plan events must be sorted with unique frames")
        if event_name in seen:
            raise ValueError(f"plan event {event_name!r} may occur only once")
        if labels is not None:
            if labels[frame_idx] != "ignore":
                raise ValueError("begin_four_tubes must be inside ignore")
            if frame_idx > 0 and labels[frame_idx - 1] == "ignore":
                raise ValueError(
                    "begin_four_tubes must mark the first frame of ignore"
                )
        previous_frame = frame_idx
        seen.add(event_name)


def save_plan_events(
    path: Path,
    document: PlanEventsDocument,
    frame_count: int,
    labels: Optional[Sequence[str]] = None,
) -> None:
    """Valida e grava ``plan_events.json``."""
    validate_plan_events(document, frame_count, labels)
    Path(path).write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_plan_events(
    path: Path,
    frame_count: int,
    labels: Optional[Sequence[str]] = None,
) -> PlanEventsDocument:
    """Carrega eventos; arquivo ausente equivale a nenhum evento."""
    event_path = Path(path)
    if not event_path.exists():
        return empty_plan_events()
    document = json.loads(event_path.read_text(encoding="utf-8"))
    validate_plan_events(document, frame_count, labels)
    return document


def toggle_begin_four_tubes(
    document: PlanEventsDocument,
    frame_idx: int,
) -> bool:
    """Move ou remove o evento ``begin_four_tubes``.

    Returns:
        ``True`` quando o evento fica marcado no quadro informado e ``False``
        quando uma marca já existente nesse quadro é removida.
    """
    events = document["events"]
    for index, event in enumerate(events):
        if event["event"] == PLAN_EVENT_BEGIN_FOUR_TUBES:
            if event["frame_idx"] == frame_idx:
                events.pop(index)
                return False
            event["frame_idx"] = frame_idx
            events.sort(key=lambda item: item["frame_idx"])
            return True
    events.append(
        {"frame_idx": frame_idx, "event": PLAN_EVENT_BEGIN_FOUR_TUBES}
    )
    events.sort(key=lambda item: item["frame_idx"])
    return True


def _load_skeleton(path: Path) -> np.ndarray:
    with path.open("rb") as skeleton_file:
        skeleton = np.asarray(pickle.load(skeleton_file), dtype=np.float32)
    if skeleton.ndim != 3 or skeleton.shape[1:] != (15, 3):
        raise ValueError(
            f"skeleton.pkl must have shape (N, 15, 3); received {skeleton.shape}"
        )
    return skeleton


class _VideoReplay:
    """Leitura aleatoria com cache de um quadro, sem carregar o video em RAM."""

    def __init__(self, path: Path) -> None:
        import cv2

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.frame_count = int(
            round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if self.frame_count <= 0:
            self.close()
            raise RuntimeError(f"video contains no frames: {path}")
        if self.fps <= 0:
            self.fps = 30.0
        self._cached_index: Optional[int] = None
        self._cached_frame: Optional[np.ndarray] = None

    def read(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"video frame index out of range: {frame_idx}")
        if self._cached_index == frame_idx and self._cached_frame is not None:
            return self._cached_frame.copy()
        if self._cached_index is None or frame_idx != self._cached_index + 1:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read video frame {frame_idx}")
        self._cached_index = frame_idx
        self._cached_frame = frame
        return frame.copy()

    def close(self) -> None:
        self._capture.release()


def draw_skeleton(
    frame: np.ndarray,
    joints15: np.ndarray,
) -> np.ndarray:
    """Desenha o esqueleto no estilo visual do ``run_webcam.py``."""
    import cv2

    display = frame.copy()
    joints = np.asarray(joints15, dtype=np.float32)
    if joints.shape != (15, 3) or not np.isfinite(joints).all():
        return display
    if np.allclose(joints, 0):
        return display

    height, width = display.shape[:2]
    scale = float(np.clip(height / 720.0, 0.8, 1.8))
    torso_thickness = max(2, int(round(4 * scale)))
    limb_thickness = max(2, int(round(3 * scale)))
    joint_radius = max(4, int(round(6 * scale)))
    points: List[Optional[Tuple[int, int]]] = []
    for x_value, y_value, _z_value in joints:
        if 0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0:
            points.append(
                (
                    int(round(float(x_value) * (width - 1))),
                    int(round(float(y_value) * (height - 1))),
                )
            )
        else:
            points.append(None)

    def draw_connections(
        connections: Sequence[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        for first, second in connections:
            if points[first] is not None and points[second] is not None:
                cv2.line(
                    display,
                    points[first],
                    points[second],
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

    draw_connections(TORSO_CONNECTIONS, (255, 80, 0), torso_thickness)
    draw_connections(ARM_CONNECTIONS, (0, 0, 255), limb_thickness)

    display_joint_indices = (0, 1, 2, 3, 4, 5, 12, 13)
    for joint_index in display_joint_indices:
        point = points[joint_index]
        if point is not None:
            cv2.circle(
                display,
                point,
                joint_radius + 2,
                (0, 70, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                display,
                point,
                joint_radius,
                (0, 220, 0),
                -1,
                cv2.LINE_AA,
            )
    return display


def _point_in_rect(point: Tuple[int, int], rect: Rect) -> bool:
    x_value, y_value = point
    left, top, right, bottom = rect
    return left <= x_value < right and top <= y_value < bottom


def _draw_button(
    canvas: np.ndarray,
    rect: Rect,
    text: str,
    *,
    background: Tuple[int, int, int] = BUTTON_BACKGROUND,
    foreground: Tuple[int, int, int] = TEXT_PRIMARY,
    selected: bool = False,
) -> None:
    import cv2

    left, top, right, bottom = rect
    fill = background
    if selected:
        fill = tuple(min(255, channel + 35) for channel in background)
    cv2.rectangle(canvas, (left, top), (right, bottom), fill, -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_BORDER, 1)
    font_scale = max(0.42, min(0.66, (bottom - top) / 65.0))
    (text_width, text_height), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        1,
    )
    text_x = left + max(5, (right - left - text_width) // 2)
    text_y = top + (bottom - top + text_height) // 2
    cv2.putText(
        canvas,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        foreground,
        1,
        cv2.LINE_AA,
    )


def render_annotation_ui(
    frame: np.ndarray,
    joints15: np.ndarray,
    *,
    frame_idx: int,
    frame_count: int,
    label: str,
    interval_start: Optional[int],
    playing: bool,
    key_bindings: Dict[str, str],
    labels: Optional[Sequence[str]] = None,
    begin_four_tubes_frame: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Rect], Rect]:
    """Compoe video, painel lateral, botoes e linha do tempo clicavel."""
    import cv2

    video = draw_skeleton(frame, joints15)
    height, width = video.shape[:2]
    panel_width = max(340, min(430, int(width * 0.32)))
    canvas = np.full(
        (height, width + panel_width, 3),
        PANEL_BACKGROUND,
        dtype=np.uint8,
    )
    canvas[:, :width] = video
    cv2.line(canvas, (width, 0), (width, height), PANEL_BORDER, 2)

    panel_left = width
    margin = 16
    content_left = panel_left + margin
    content_right = width + panel_width - margin
    inner_width = content_right - content_left
    scale = float(np.clip(height / 720.0, 0.72, 1.25))

    def put(
        text: str,
        y_value: int,
        color: Tuple[int, int, int] = TEXT_PRIMARY,
        font_scale: float = 0.58,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            canvas,
            text,
            (content_left, y_value),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    put("HRC ANNOTATION", int(34 * scale), (100, 255, 100), 0.72, 2)
    put(f"Frame {frame_idx + 1} / {frame_count}", int(67 * scale), thickness=2)
    put(f"Classe: {label}", int(96 * scale), LABEL_COLORS[label], 0.58, 2)
    selection = (
        f"Inicio marcado: {interval_start + 1}"
        if interval_start is not None
        else "Inicio marcado: nenhum"
    )
    put(selection, int(122 * scale), TEXT_SECONDARY, 0.5)
    event_text = (
        f"four_tubes: frame {begin_four_tubes_frame + 1}"
        if begin_four_tubes_frame is not None
        else "four_tubes: nao marcado"
    )
    put(event_text, int(139 * scale), (0, 190, 255), 0.45)

    timeline_top = int(153 * scale)
    timeline_rect = (
        content_left,
        timeline_top,
        content_right,
        timeline_top + max(16, int(20 * scale)),
    )
    timeline_width = timeline_rect[2] - timeline_rect[0]
    cv2.rectangle(
        canvas,
        (timeline_rect[0], timeline_rect[1]),
        (timeline_rect[2], timeline_rect[3]),
        (60, 60, 68),
        -1,
    )
    timeline_labels = labels if labels is not None else [label] * frame_count
    if len(timeline_labels) != frame_count:
        raise ValueError("labels length must match frame_count")
    run_start = 0
    run_label = timeline_labels[0]
    for index in range(1, frame_count + 1):
        if index == frame_count or timeline_labels[index] != run_label:
            left = timeline_rect[0] + int(run_start / frame_count * timeline_width)
            right = timeline_rect[0] + int(index / frame_count * timeline_width)
            cv2.rectangle(
                canvas,
                (left, timeline_rect[1]),
                (max(left + 1, right), timeline_rect[3]),
                LABEL_COLORS[run_label],
                -1,
            )
            if index < frame_count:
                run_start = index
                run_label = timeline_labels[index]

    progress = 0.0 if frame_count <= 1 else frame_idx / (frame_count - 1)
    progress_x = timeline_rect[0] + int(progress * timeline_width)
    cv2.line(
        canvas,
        (progress_x, timeline_rect[1] - 5),
        (progress_x, timeline_rect[3] + 5),
        (255, 255, 255),
        2,
    )
    if interval_start is not None and frame_count > 1:
        mark_x = timeline_rect[0] + int(
            interval_start
            / (frame_count - 1)
            * (timeline_rect[2] - timeline_rect[0])
        )
        cv2.line(
            canvas,
            (mark_x, timeline_rect[1] - 4),
            (mark_x, timeline_rect[3] + 4),
            (255, 255, 255),
            2,
        )

    buttons: Dict[str, Rect] = {}
    gap = 6
    nav_top = timeline_rect[3] + int(18 * scale)
    nav_height = max(32, int(42 * scale))
    nav_width = (inner_width - gap * 4) // 5
    nav_items = (
        ("back10", "<<10"),
        ("back1", "<1"),
        ("play", "PAUSE" if playing else "PLAY"),
        ("forward1", "1>"),
        ("forward10", "10>>"),
    )
    for index, (action, text) in enumerate(nav_items):
        left = content_left + index * (nav_width + gap)
        rect = (left, nav_top, left + nav_width, nav_top + nav_height)
        buttons[action] = rect
        _draw_button(
            canvas,
            rect,
            text,
            selected=action == "play" and playing,
        )

    utility_top = nav_top + nav_height + int(10 * scale)
    utility_gap = 6
    utility_width = (inner_width - utility_gap * 2) // 3
    utility_height = max(34, int(44 * scale))
    for index, (action, text) in enumerate(
        (
            (
                "mark",
                "CANCELAR MARCA"
                if interval_start is not None
                else "MARCAR INICIO",
            ),
            ("four_tubes", "F FOUR_TUBES"),
            ("undo", "DESFAZER"),
        )
    ):
        left = content_left + index * (utility_width + utility_gap)
        rect = (
            left,
            utility_top,
            left + utility_width,
            utility_top + utility_height,
        )
        buttons[action] = rect
        _draw_button(
            canvas,
            rect,
            text,
            selected=action == "mark" and interval_start is not None,
        )

    classes_title_y = utility_top + utility_height + int(22 * scale)
    put("Aplicar classe", classes_title_y, TEXT_SECONDARY, 0.48)
    classes_top = classes_title_y + int(12 * scale)
    footer_height = max(40, int(48 * scale))
    footer_top = height - footer_height - margin
    available_height = footer_top - classes_top - int(14 * scale)
    class_gap = max(3, int(5 * scale))
    class_height = max(
        28,
        min(int(48 * scale), (available_height - class_gap * 4) // 5),
    )
    for index, class_name in enumerate(ANNOTATION_LABELS):
        top = classes_top + index * (class_height + class_gap)
        rect = (content_left, top, content_right, top + class_height)
        buttons[f"class:{class_name}"] = rect
        blue, green, red = LABEL_COLORS[class_name]
        luminance = 0.114 * blue + 0.587 * green + 0.299 * red
        foreground = (20, 20, 20) if luminance > 145 else (255, 255, 255)
        class_text = (
            f"{key_bindings[class_name]}   DESMARCAR / NO ACTION"
            if class_name == "no_action"
            else f"{key_bindings[class_name]}   {class_name}"
        )
        _draw_button(
            canvas,
            rect,
            class_text,
            background=LABEL_COLORS[class_name],
            foreground=foreground,
            selected=class_name == label,
        )

    footer_gap = 8
    footer_width = (inner_width - footer_gap) // 2
    save_rect = (
        content_left,
        footer_top,
        content_left + footer_width,
        footer_top + footer_height,
    )
    cancel_rect = (
        content_left + footer_width + footer_gap,
        footer_top,
        content_right,
        footer_top + footer_height,
    )
    buttons["save"] = save_rect
    buttons["cancel"] = cancel_rect
    _draw_button(canvas, save_rect, "SALVAR", background=(30, 125, 65))
    _draw_button(canvas, cancel_rect, "CANCELAR", background=(80, 60, 60))
    return canvas, buttons, timeline_rect


def action_at_point(
    point: Tuple[int, int],
    buttons: Dict[str, Rect],
    timeline_rect: Rect,
    frame_count: int,
) -> Optional[Tuple[str, Optional[int]]]:
    """Traduz clique em uma acao de botao ou salto na linha do tempo."""
    for action, rect in buttons.items():
        if _point_in_rect(point, rect):
            return action, None
    if _point_in_rect(point, timeline_rect):
        width = max(1, timeline_rect[2] - timeline_rect[0] - 1)
        ratio = (point[0] - timeline_rect[0]) / width
        frame_idx = int(round(ratio * max(0, frame_count - 1)))
        return "seek", min(max(frame_idx, 0), max(0, frame_count - 1))
    return None


def _normalized_key_bindings(
    key_bindings: Optional[Dict[str, str]],
) -> Dict[str, str]:
    bindings = dict(DEFAULT_KEY_BINDINGS)
    if key_bindings is not None:
        unknown = set(key_bindings).difference(ANNOTATION_LABELS)
        if unknown:
            raise ValueError(f"key bindings contain unknown labels: {sorted(unknown)}")
        bindings.update(key_bindings)
    if any(not isinstance(key, str) or len(key) != 1 for key in bindings.values()):
        raise ValueError("every annotation key binding must be one character")
    if len(set(bindings.values())) != len(bindings):
        raise ValueError("annotation key bindings must be unique")
    reserved = {"b", "f", "u", "s", "q", "a", "d", "j", "l", " "}
    collision = reserved.intersection(bindings.values())
    if collision:
        raise ValueError(f"annotation keys collide with controls: {sorted(collision)}")
    return bindings


def annotate_session(
    session_dir: Path,
    key_bindings: Optional[Dict[str, str]] = None,
) -> Path:
    """Abre o replay sincronizado e grava ``annotations.json``.

    Args:
        session_dir: Diretorio com ``video.mp4`` e ``skeleton.pkl``.
        key_bindings: Mapa opcional de tecla para cada rotulo em
            ``ANNOTATION_LABELS``.

    Returns:
        Caminho do ``annotations.json`` salvo.

    Notes:
        ``no_action`` cobre inicialmente todos os quadros. Pressione ``b`` no
        inicio, navegue ate o fim e pressione a tecla da classe para aplicar o
        intervalo inclusivo. ``ignore`` e explicito e nao possui ID de classe.
    """
    import cv2

    session_dir = Path(session_dir)
    skeleton_path = session_dir / "skeleton.pkl"
    video_path = session_dir / "video.mp4"
    annotations_path = session_dir / "annotations.json"
    plan_events_path = session_dir / "plan_events.json"
    if not skeleton_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(
            f"session must contain skeleton.pkl and video.mp4: {session_dir}"
        )

    bindings = _normalized_key_bindings(key_bindings)
    key_to_label = {key: label for label, key in bindings.items()}
    skeleton = _load_skeleton(skeleton_path)
    replay = _VideoReplay(video_path)
    try:
        if replay.frame_count != len(skeleton):
            raise ValueError(
                "video and skeleton frame counts differ: "
                f"video={replay.frame_count}, skeleton={len(skeleton)}"
            )

        frame_count = replay.frame_count
        if annotations_path.exists():
            labels = annotations_to_labels(
                load_annotations(annotations_path, frame_count),
                frame_count,
            )
        else:
            labels = ["no_action"] * frame_count
        plan_events = load_plan_events(
            plan_events_path,
            frame_count,
            labels if annotations_path.exists() else None,
        )

        frame_idx = 0
        interval_start: Optional[int] = None
        playing = False
        undo_stack: List[Tuple[int, int, List[str]]] = []
        delay_ms = max(1, int(round(1000.0 / replay.fps)))
        window_name = "HRC Annotation"
        ui_buttons: Dict[str, Rect] = {}
        ui_timeline: Rect = (0, 0, 0, 0)
        pending_clicks: List[Tuple[int, int]] = []

        def on_mouse(
            event: int,
            x_value: int,
            y_value: int,
            _flags: int,
            _userdata: Any,
        ) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_clicks.append((x_value, y_value))

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            frame = replay.read(frame_idx)
            display, ui_buttons, ui_timeline = render_annotation_ui(
                frame,
                skeleton[frame_idx],
                frame_idx=frame_idx,
                frame_count=frame_count,
                label=labels[frame_idx],
                interval_start=interval_start,
                playing=playing,
                key_bindings=bindings,
                labels=labels,
                begin_four_tubes_frame=(
                    plan_events["events"][0]["frame_idx"]
                    if plan_events["events"]
                    else None
                ),
            )
            cv2.imshow(window_name, display)
            key_code = cv2.waitKeyEx(delay_ms if playing else 30)

            click_action = None
            if pending_clicks:
                click_action = action_at_point(
                    pending_clicks.pop(0),
                    ui_buttons,
                    ui_timeline,
                    frame_count,
                )

            if key_code == -1 and click_action is None:
                if playing:
                    if frame_idx < frame_count - 1:
                        frame_idx += 1
                    else:
                        playing = False
                continue

            low_byte = key_code & 0xFF
            key = chr(low_byte).lower() if 0 <= low_byte < 128 else ""
            if click_action is not None:
                action, value = click_action
                if action == "seek" and value is not None:
                    frame_idx = value
                    playing = False
                    continue
                if action == "back10":
                    frame_idx = max(0, frame_idx - 10)
                    playing = False
                    continue
                if action == "back1":
                    frame_idx = max(0, frame_idx - 1)
                    playing = False
                    continue
                if action == "forward1":
                    frame_idx = min(frame_count - 1, frame_idx + 1)
                    playing = False
                    continue
                if action == "forward10":
                    frame_idx = min(frame_count - 1, frame_idx + 10)
                    playing = False
                    continue
                if action == "play":
                    key = " "
                elif action == "mark":
                    key = "b"
                elif action == "undo":
                    key = "u"
                elif action == "four_tubes":
                    key = "f"
                elif action == "save":
                    key = "s"
                elif action == "cancel":
                    key = "q"
                elif action.startswith("class:"):
                    key = bindings[action.split(":", 1)[1]]

            if key == " ":
                playing = not playing
            elif key == "b":
                interval_start = (
                    None if interval_start is not None else frame_idx
                )
                playing = False
            elif key in key_to_label:
                start = (
                    interval_start
                    if interval_start is not None
                    else frame_idx
                )
                interval_start, interval_end = sorted((start, frame_idx))
                undo_stack.append(
                    (
                        interval_start,
                        interval_end,
                        labels[interval_start : interval_end + 1].copy(),
                    )
                )
                assign_interval(
                    labels,
                    interval_start,
                    interval_end,
                    key_to_label[key],
                )
                interval_start = None
                playing = False
            elif key == "u" and undo_stack:
                start, end, previous_labels = undo_stack.pop()
                labels[start : end + 1] = previous_labels
                frame_idx = start
                interval_start = None
                playing = False
            elif key == "f":
                toggle_begin_four_tubes(plan_events, frame_idx)
                playing = False
            elif key == "a":
                frame_idx = max(0, frame_idx - 1)
                playing = False
            elif key == "d":
                frame_idx = min(frame_count - 1, frame_idx + 1)
                playing = False
            elif key == "j":
                frame_idx = max(0, frame_idx - 10)
                playing = False
            elif key == "l":
                frame_idx = min(frame_count - 1, frame_idx + 10)
                playing = False
            elif key_code in (81, 2424832):
                frame_idx = max(0, frame_idx - 1)
                playing = False
            elif key_code in (83, 2555904):
                frame_idx = min(frame_count - 1, frame_idx + 1)
                playing = False
            elif key == "s":
                annotations = labels_to_annotations(labels)
                save_annotations(annotations_path, annotations, frame_count)
                save_plan_events(
                    plan_events_path,
                    plan_events,
                    frame_count,
                    labels,
                )
                return annotations_path
            elif key == "q" or key_code == 27:
                raise RuntimeError("annotation cancelled without saving")

            if playing and frame_idx < frame_count - 1:
                frame_idx += 1
            elif playing:
                playing = False
    finally:
        replay.close()
        cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Executa a interface de linha de comando do anotador."""
    parser = argparse.ArgumentParser(
        description="Annotate a captured HRC session frame by frame."
    )
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        annotations_path = annotate_session(args.session_dir)
    except RuntimeError as exc:
        if str(exc) == "annotation cancelled without saving":
            print(exc)
            return 1
        raise
    print(annotations_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
