"""Replay grafico para auditar contexto, anotacoes e estado do PlanGraph."""

import argparse
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datacol.annotate_pkl import (
    LABEL_COLORS,
    PANEL_BACKGROUND,
    PANEL_BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _VideoReplay,
    _load_skeleton,
    annotations_to_labels,
    draw_skeleton,
    load_annotations,
    load_plan_events,
)
from datacol.build_json import build_plan_timeline

Rect = Tuple[int, int, int, int]

CONTEXT_NAMES = {
    7: (
        "stage:none",
        "stage:bottom",
        "stage:four_tubes",
        "stage:top",
        "connectors / 8",
        "screws / 12",
        "wheels / 4",
    ),
    10: (
        "stage:none",
        "stage:bottom",
        "stage:four_tubes",
        "stage:top",
        "short tubes / 8",
        "long tubes / 4",
        "screws bottom / 4",
        "screws four_tubes / 4",
        "screws top / 4",
        "wheels / 4",
    ),
}

STAGE_COLORS = {
    "bottom": (80, 180, 255),
    "four_tubes": (90, 220, 130),
    "top": (220, 150, 80),
}

BUTTON_BACKGROUND = (52, 52, 60)
BUTTON_ACTIVE = (55, 135, 75)


def load_context_replay(
    session_dir: Path,
    context_dim: int = 7,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """Carrega esqueleto, rotulos e estado do plano por quadro."""
    session = Path(session_dir)
    skeleton = _load_skeleton(session / "skeleton.pkl")
    labels = annotations_to_labels(
        load_annotations(session / "annotations.json", len(skeleton)),
        len(skeleton),
    )
    events = load_plan_events(
        session / "plan_events.json",
        len(skeleton),
        labels,
    )
    timeline = build_plan_timeline(labels, context_dim, events)
    return skeleton, labels, timeline


def _put_text(
    canvas: np.ndarray,
    text: str,
    point: Tuple[int, int],
    *,
    color: Tuple[int, int, int] = TEXT_PRIMARY,
    scale: float = 0.48,
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        canvas,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_context_bars(
    canvas: np.ndarray,
    rect: Rect,
    context: Sequence[float],
) -> None:
    import cv2

    left, top, right, bottom = rect
    names = CONTEXT_NAMES[len(context)]
    row_height = max(18, (bottom - top) // len(context))
    label_width = 150
    value_width = 42
    bar_left = left + label_width
    bar_right = right - value_width

    for index, (name, value) in enumerate(zip(names, context)):
        y_value = top + index * row_height
        _put_text(
            canvas,
            name,
            (left, y_value + 14),
            color=TEXT_SECONDARY,
            scale=0.38,
        )
        cv2.rectangle(
            canvas,
            (bar_left, y_value + 3),
            (bar_right, y_value + 15),
            (55, 55, 62),
            -1,
        )
        fill_right = bar_left + int(
            np.clip(value, 0.0, 1.0) * (bar_right - bar_left)
        )
        cv2.rectangle(
            canvas,
            (bar_left, y_value + 3),
            (fill_right, y_value + 15),
            (80, 210, 120),
            -1,
        )
        _put_text(
            canvas,
            f"{value:.3f}",
            (bar_right + 6, y_value + 14),
            scale=0.36,
        )


def _draw_button(
    canvas: np.ndarray,
    rect: Rect,
    text: str,
    *,
    active: bool = False,
) -> None:
    import cv2

    left, top, right, bottom = rect
    background = BUTTON_ACTIVE if active else BUTTON_BACKGROUND
    cv2.rectangle(canvas, (left, top), (right, bottom), background, -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_BORDER, 1)
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1,
    )
    _put_text(
        canvas,
        text,
        (
            left + max(4, (right - left - text_width) // 2),
            top + (bottom - top + text_height) // 2,
        ),
        scale=0.38,
    )


def draw_proxy_diagram(
    canvas: np.ndarray,
    rect: Rect,
    snapshot: Dict[str, Any],
) -> None:
    """Desenha uma representacao esquematica do progresso da montagem."""
    import cv2

    left, top, right, bottom = rect
    cv2.rectangle(canvas, (left, top), (right, bottom), (34, 34, 40), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_BORDER, 1)
    _put_text(
        canvas,
        "PROXY ASSEMBLY",
        (left + 12, top + 22),
        color=(100, 255, 100),
        scale=0.48,
        thickness=2,
    )

    stage_record = snapshot["stage_record"]
    screw_count = snapshot["screw_count"]
    active_stage = snapshot["stage"]
    empty_color = (62, 62, 70)

    car_left = left + 72
    car_right = right - 34
    car_top = top + 55
    car_bottom = bottom - 48
    upper_y = car_top + 22
    lower_y = car_bottom - 34
    front_x = car_right - 26
    rear_x = car_left + 26
    post_x = (
        rear_x,
        rear_x + (front_x - rear_x) // 3,
        rear_x + 2 * (front_x - rear_x) // 3,
        front_x,
    )

    def stage_color(stage: str, complete: bool) -> Tuple[int, int, int]:
        if complete:
            return STAGE_COLORS[stage]
        if active_stage == stage:
            return tuple(channel // 2 for channel in STAGE_COLORS[stage])
        return empty_color

    bottom_count = min(4, len(stage_record["bottom"]))
    top_count = min(4, len(stage_record["top"]))
    post_count = min(4, len(stage_record["four_tubes"]))
    chassis_width = car_right - car_left
    bottom_fill = car_left + int(chassis_width * bottom_count / 4)
    top_fill = car_left + int(chassis_width * top_count / 4)

    # Trilhos inferior e superior do carrinho.
    cv2.line(canvas, (car_left, lower_y), (car_right, lower_y), empty_color, 12)
    cv2.line(
        canvas,
        (car_left, lower_y),
        (bottom_fill, lower_y),
        stage_color("bottom", bottom_count == 4),
        12,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (car_left, upper_y), (car_right, upper_y), empty_color, 12)
    cv2.line(
        canvas,
        (car_left, upper_y),
        (top_fill, upper_y),
        stage_color("top", top_count == 4),
        12,
        cv2.LINE_AA,
    )

    # Quatro colunas centrais representam o estágio four_tubes.
    for index, x_value in enumerate(post_x):
        color = stage_color("four_tubes", index < post_count)
        cv2.line(
            canvas,
            (x_value, upper_y + 7),
            (x_value, lower_y - 7),
            color,
            9,
            cv2.LINE_AA,
        )

    # Parafusos aparecem nos pontos de união de cada estágio.
    screw_positions = {
        "top": [(x_value, upper_y) for x_value in post_x],
        "four_tubes": [
            (x_value, (upper_y + lower_y) // 2) for x_value in post_x
        ],
        "bottom": [(x_value, lower_y) for x_value in post_x],
    }
    for stage, positions in screw_positions.items():
        completed = min(4, screw_count[stage])
        for index, point in enumerate(positions):
            fill = (245, 245, 245) if index < completed else (82, 82, 90)
            cv2.circle(canvas, point, 6, fill, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 6, STAGE_COLORS[stage], 1, cv2.LINE_AA)

    # Rodas em perspectiva: duas dianteiras maiores e duas traseiras menores.
    wheel_count = min(4, snapshot["wheels_count"])
    wheel_positions = (
        (rear_x, lower_y + 31, 18),
        (front_x, lower_y + 31, 18),
        (rear_x + 35, lower_y + 20, 12),
        (front_x - 35, lower_y + 20, 12),
    )
    for index, (x_value, y_value, radius) in enumerate(wheel_positions):
        fill = (
            LABEL_COLORS["get_wheels"] if index < wheel_count else empty_color
        )
        cv2.circle(canvas, (x_value, y_value), radius, fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x_value, y_value), radius, (210, 210, 215), 2)
        cv2.circle(canvas, (x_value, y_value), max(3, radius // 3), (28, 28, 32), -1)

    labels = (
        ("TOP", top_count, screw_count["top"], upper_y),
        ("FOUR TUBES", post_count, screw_count["four_tubes"], (upper_y + lower_y) // 2),
        ("BOTTOM", bottom_count, screw_count["bottom"], lower_y),
    )
    for name, pieces, screws, y_value in labels:
        stage_key = name.lower().replace(" ", "_")
        _put_text(
            canvas,
            name,
            (left + 8, y_value + 4),
            color=STAGE_COLORS[stage_key],
            scale=0.31,
            thickness=1,
        )
        _put_text(
            canvas,
            f"{pieces}/4  S:{min(4, screws)}/4",
            (right - 89, y_value - 12),
            color=TEXT_SECONDARY,
            scale=0.29,
        )
    _put_text(
        canvas,
        f"WHEELS {wheel_count}/4",
        (right - 103, bottom - 8),
        color=TEXT_SECONDARY,
        scale=0.31,
    )


def render_context_ui(
    frame: np.ndarray,
    joints15: np.ndarray,
    *,
    frame_idx: int,
    frame_count: int,
    label: str,
    labels: Sequence[str],
    frame_state: Dict[str, Any],
    playing: bool = False,
    exporting: bool = False,
) -> Tuple[np.ndarray, Dict[str, Rect], Rect]:
    """Compoe o replay, a classificacao, o contexto e o desenho da proxy."""
    import cv2

    height, width = frame.shape[:2]
    target_height = min(height, 720)
    target_width = max(1, int(round(width * target_height / height)))
    resized = cv2.resize(frame, (target_width, target_height))
    video = draw_skeleton(resized, joints15)

    panel_width = 520
    canvas = np.full(
        (target_height, target_width + panel_width, 3),
        PANEL_BACKGROUND,
        dtype=np.uint8,
    )
    canvas[:, :target_width] = video
    cv2.line(
        canvas,
        (target_width, 0),
        (target_width, target_height),
        PANEL_BORDER,
        2,
    )
    left = target_width + 16
    right = target_width + panel_width - 16
    snapshot = frame_state["snapshot"]
    context = frame_state["context"]
    stage = snapshot["stage"] or "none"

    _put_text(
        canvas,
        "PLAN CONTEXT AUDIT",
        (left, 28),
        color=(100, 255, 100),
        scale=0.62,
        thickness=2,
    )
    _put_text(
        canvas,
        f"Frame {frame_idx + 1}/{frame_count}  "
        f"{'PLAY' if playing else 'PAUSE'}",
        (left, 55),
        scale=0.48,
        thickness=2,
    )
    _put_text(
        canvas,
        f"Action: {label}",
        (left, 80),
        color=LABEL_COLORS[label],
        scale=0.52,
        thickness=2,
    )
    _put_text(
        canvas,
        f"Active stage: {stage}",
        (left, 103),
        color=STAGE_COLORS.get(stage, TEXT_SECONDARY),
        scale=0.44,
    )
    if frame_state["event"] is not None:
        _put_text(
            canvas,
            f"Event: {frame_state['event']}",
            (left + 220, 103),
            color=(0, 190, 255),
            scale=0.40,
            thickness=2,
        )

    bars_bottom = 250 if len(context) == 7 else 292
    _draw_context_bars(canvas, (left, 116, right, bars_bottom), context)
    diagram_top = bars_bottom + 10
    controls_top = target_height - 76
    timeline_top = target_height - 42
    draw_proxy_diagram(
        canvas,
        (left, diagram_top, right, controls_top - 10),
        snapshot,
    )

    buttons: Dict[str, Rect] = {}
    button_gap = 5
    button_items = (
        ("back10", "-10"),
        ("back1", "-1"),
        ("play", "PAUSE" if playing else "PLAY"),
        ("forward1", "+1"),
        ("forward10", "+10"),
        ("save", "SAVING..." if exporting else "SAVE MP4"),
    )
    button_width = (right - left - button_gap * (len(button_items) - 1)) // len(
        button_items
    )
    for index, (action, text) in enumerate(button_items):
        button_left = left + index * (button_width + button_gap)
        rect = (
            button_left,
            controls_top,
            button_left + button_width,
            controls_top + 27,
        )
        buttons[action] = rect
        _draw_button(
            canvas,
            rect,
            text,
            active=(action == "play" and playing) or (
                action == "save" and exporting
            ),
        )

    timeline_rect = (left, timeline_top, right, timeline_top + 16)
    timeline_width = right - left
    run_start = 0
    run_label = labels[0]
    for index in range(1, frame_count + 1):
        if index == frame_count or labels[index] != run_label:
            segment_left = left + int(run_start / frame_count * timeline_width)
            segment_right = left + int(index / frame_count * timeline_width)
            cv2.rectangle(
                canvas,
                (segment_left, timeline_top),
                (max(segment_left + 1, segment_right), timeline_top + 16),
                LABEL_COLORS[run_label],
                -1,
            )
            if index < frame_count:
                run_start = index
                run_label = labels[index]
    progress = 0.0 if frame_count <= 1 else frame_idx / (frame_count - 1)
    progress_x = left + int(progress * timeline_width)
    cv2.line(
        canvas,
        (progress_x, timeline_top - 4),
        (progress_x, timeline_top + 20),
        (255, 255, 255),
        2,
    )
    _put_text(
        canvas,
        "Space play | A/D 1 | J/L 10 | E save MP4 | Q exit",
        (left, target_height - 9),
        color=TEXT_SECONDARY,
        scale=0.32,
    )
    return canvas, buttons, timeline_rect


def action_at_point(
    point: Tuple[int, int],
    buttons: Dict[str, Rect],
    timeline_rect: Rect,
    frame_count: int,
) -> Optional[Tuple[str, Optional[int]]]:
    """Converte um clique em controle ou posição da linha do tempo."""
    x_value, y_value = point
    for action, (left, top, right, bottom) in buttons.items():
        if left <= x_value < right and top <= y_value < bottom:
            return action, None
    left, top, right, bottom = timeline_rect
    if left <= x_value < right and top <= y_value < bottom:
        ratio = (x_value - left) / max(1, right - left - 1)
        frame_idx = int(round(ratio * max(0, frame_count - 1)))
        return "seek", min(max(frame_idx, 0), max(0, frame_count - 1))
    return None


def export_context_video(
    session_dir: Path,
    output_path: Path,
    context_dim: int = 7,
) -> Path:
    """Renderiza todo o replay, incluindo painel e contexto, em um MP4."""
    import cv2

    session = Path(session_dir)
    output = Path(output_path)
    skeleton, labels, timeline = load_context_replay(session, context_dim)
    replay = _VideoReplay(session / "video.mp4")
    writer = None
    ffmpeg_process = None
    try:
        if replay.frame_count != len(skeleton):
            raise ValueError(
                "video and skeleton frame counts differ: "
                f"video={replay.frame_count}, skeleton={len(skeleton)}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(replay.frame_count):
            canvas, _buttons, _timeline_rect = render_context_ui(
                replay.read(frame_idx),
                skeleton[frame_idx],
                frame_idx=frame_idx,
                frame_count=replay.frame_count,
                label=labels[frame_idx],
                labels=labels,
                frame_state=timeline[frame_idx],
                playing=True,
                exporting=True,
            )
            if writer is None:
                height, width = canvas.shape[:2]
                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path is not None:
                    ffmpeg_process = subprocess.Popen(
                        [
                            ffmpeg_path,
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "bgr24",
                            "-s:v",
                            f"{width}x{height}",
                            "-r",
                            f"{replay.fps:.6f}",
                            "-i",
                            "-",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "medium",
                            "-crf",
                            "18",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(output),
                        ],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    writer = ffmpeg_process.stdin
                else:
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        replay.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"could not create output video: {output}"
                        )
            if ffmpeg_process is not None:
                writer.write(canvas.tobytes())
            else:
                writer.write(canvas)
    finally:
        if writer is not None:
            if ffmpeg_process is not None:
                writer.close()
                error_output = ffmpeg_process.stderr.read().decode(
                    "utf-8",
                    errors="replace",
                )
                return_code = ffmpeg_process.wait()
                if return_code != 0:
                    raise RuntimeError(
                        "ffmpeg could not encode output video: "
                        f"{error_output.strip()}"
                    )
            else:
                writer.release()
        replay.close()
    return output


def replay_context(
    session_dir: Path,
    context_dim: int = 7,
    start_frame: int = 0,
    autoplay: bool = True,
    export_path: Optional[Path] = None,
) -> None:
    """Abre a auditoria grafica sincronizada de uma sessao anotada."""
    import cv2

    session = Path(session_dir)
    skeleton, labels, timeline = load_context_replay(session, context_dim)
    replay = _VideoReplay(session / "video.mp4")
    if replay.frame_count != len(skeleton):
        replay.close()
        raise ValueError(
            "video and skeleton frame counts differ: "
            f"video={replay.frame_count}, skeleton={len(skeleton)}"
        )
    if start_frame < 0 or start_frame >= replay.frame_count:
        replay.close()
        raise ValueError("start_frame is outside the session")

    frame_idx = start_frame
    playing = autoplay
    delay_ms = max(1, int(round(1000.0 / replay.fps)))
    window_name = "HRC Plan Context Audit"
    pending_clicks: List[Tuple[int, int]] = []
    buttons: Dict[str, Rect] = {}
    timeline_rect: Rect = (0, 0, 0, 0)
    output_path = (
        Path(export_path)
        if export_path is not None
        else session / "context_replay.mp4"
    )

    def on_mouse(
        event: int,
        x_value: int,
        y_value: int,
        _flags: int,
        _userdata: Any,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_clicks.append((x_value, y_value))

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)
        while True:
            frame = replay.read(frame_idx)
            canvas, buttons, timeline_rect = render_context_ui(
                frame,
                skeleton[frame_idx],
                frame_idx=frame_idx,
                frame_count=replay.frame_count,
                label=labels[frame_idx],
                labels=labels,
                frame_state=timeline[frame_idx],
                playing=playing,
            )
            cv2.imshow(window_name, canvas)
            key_code = cv2.waitKeyEx(delay_ms if playing else 30)

            click_action = None
            if pending_clicks:
                click_action = action_at_point(
                    pending_clicks.pop(0),
                    buttons,
                    timeline_rect,
                    replay.frame_count,
                )
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
                        frame_idx = min(replay.frame_count - 1, frame_idx + 1)
                        playing = False
                        continue
                    if action == "forward10":
                        frame_idx = min(replay.frame_count - 1, frame_idx + 10)
                        playing = False
                        continue
                    if action == "play":
                        playing = not playing
                        continue
                    if action == "save":
                        playing = False
                        print(f"exporting: {output_path}")
                        export_context_video(session, output_path, context_dim)
                        print(f"saved: {output_path}")
                        continue

            if key_code == -1:
                if playing:
                    if frame_idx < replay.frame_count - 1:
                        frame_idx += 1
                    else:
                        playing = False
                continue

            low_byte = key_code & 0xFF
            key = chr(low_byte).lower() if 0 <= low_byte < 128 else ""
            if key == " ":
                playing = not playing
            elif key == "a" or key_code in (81, 2424832):
                frame_idx = max(0, frame_idx - 1)
                playing = False
            elif key == "d" or key_code in (83, 2555904):
                frame_idx = min(replay.frame_count - 1, frame_idx + 1)
                playing = False
            elif key == "j":
                frame_idx = max(0, frame_idx - 10)
                playing = False
            elif key == "l":
                frame_idx = min(replay.frame_count - 1, frame_idx + 10)
                playing = False
            elif key == "e":
                playing = False
                print(f"exporting: {output_path}")
                export_context_video(session, output_path, context_dim)
                print(f"saved: {output_path}")
            elif key == "q" or key_code == 27:
                return

            if playing and frame_idx < replay.frame_count - 1:
                frame_idx += 1
            elif playing:
                playing = False
    finally:
        replay.close()
        cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Executa a interface de auditoria do contexto."""
    parser = argparse.ArgumentParser(
        description="Replay grafico do video, anotacoes e PlanGraph.",
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--context-dim", type=int, choices=(7, 10), default=7)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--paused",
        action="store_true",
        help="abre no quadro inicial sem iniciar a reproducao",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help=(
            "caminho usado pelo botao SAVE MP4; por padrao grava "
            "context_replay.mp4 dentro da sessao"
        ),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="gera o MP4 e encerra sem abrir a interface",
    )
    args = parser.parse_args(argv)
    if args.export_only:
        output = args.output_video or args.session_dir / "context_replay.mp4"
        exported = export_context_video(
            args.session_dir,
            output,
            args.context_dim,
        )
        print(exported)
        return 0
    replay_context(
        args.session_dir,
        args.context_dim,
        args.start_frame,
        autoplay=not args.paused,
        export_path=args.output_video,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
