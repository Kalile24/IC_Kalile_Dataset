"""Criterios de aceite do anotador de intervalos."""

import json
from pathlib import Path

import cv2
from jsonschema import Draft202012Validator
import numpy as np
import pytest

from datacol import ANNOTATION_LABELS, INTENTION_LIST
from datacol.annotate_pkl import (
    DEFAULT_KEY_BINDINGS,
    action_at_point,
    annotations_to_labels,
    assign_interval,
    draw_skeleton,
    empty_plan_events,
    labels_to_annotations,
    load_annotations,
    render_annotation_ui,
    save_annotations,
    save_plan_events,
    load_plan_events,
    toggle_begin_four_tubes,
    validate_annotations,
    validate_plan_events,
)


def test_annotations_cover_every_frame_exactly_once() -> None:
    """Cada quadro deve ter um rotulo, usando no_action como default."""
    labels = ["no_action"] * 12
    assign_interval(labels, 2, 4, "get_connectors")
    assign_interval(labels, 6, 8, "get_screws")
    assign_interval(labels, 10, 10, "get_wheels")

    annotations = labels_to_annotations(labels)
    validate_annotations(annotations, frame_count=len(labels))

    assert annotations_to_labels(annotations, len(labels)) == labels
    covered = sum(
        end - start + 1
        for label in ANNOTATION_LABELS
        for start, end in zip(
            annotations[label]["start"],
            annotations[label]["end"],
        )
    )
    assert covered == len(labels)


def test_annotation_intervals_do_not_overlap() -> None:
    """Intervalos inclusivos de classes distintas nao podem se sobrepor."""
    annotations = {
        label: {"start": [], "end": []}
        for label in ANNOTATION_LABELS
    }
    annotations["no_action"] = {"start": [0], "end": [5]}
    annotations["get_connectors"] = {"start": [5], "end": [9]}

    with pytest.raises(ValueError, match="overlaps"):
        validate_annotations(annotations, frame_count=10)


def test_annotation_gaps_are_rejected() -> None:
    """Todo quadro deve estar coberto; lacunas nao podem ser salvas."""
    annotations = {
        label: {"start": [], "end": []}
        for label in ANNOTATION_LABELS
    }
    annotations["no_action"] = {"start": [0, 6], "end": [3, 9]}

    with pytest.raises(ValueError, match="not fully covered"):
        validate_annotations(annotations, frame_count=10)


def test_ignore_is_explicit_and_has_no_class_id() -> None:
    """ignore deve ser salvo, mas nao integrar INTENTION_LIST."""
    labels = ["no_action"] * 8
    assign_interval(labels, 3, 5, "ignore")
    annotations = labels_to_annotations(labels)

    assert "ignore" in annotations
    assert annotations["ignore"] == {"start": [3], "end": [5]}
    assert "ignore" not in INTENTION_LIST


def test_annotations_round_trip_without_loss(tmp_path: Path) -> None:
    """Salvar e recarregar annotations.json deve preservar todos os intervalos."""
    labels = [
        "no_action",
        "get_connectors",
        "get_connectors",
        "ignore",
        "get_screws",
        "get_wheels",
        "no_action",
    ]
    annotations = labels_to_annotations(labels)
    path = tmp_path / "annotations.json"

    save_annotations(path, annotations, frame_count=len(labels))
    loaded = load_annotations(path, frame_count=len(labels))

    assert loaded == annotations
    assert annotations_to_labels(loaded, len(labels)) == labels
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "annotations.schema.json"
    )
    Draft202012Validator(
        json.loads(schema_path.read_text())
    ).validate(loaded)


def test_assign_interval_accepts_reverse_navigation() -> None:
    """Marcar o fim antes do inicio deve produzir o mesmo intervalo inclusivo."""
    labels = ["no_action"] * 6

    bounds = assign_interval(labels, 4, 2, "get_wheels")

    assert bounds == (2, 4)
    assert labels == [
        "no_action",
        "no_action",
        "get_wheels",
        "get_wheels",
        "get_wheels",
        "no_action",
    ]


def test_skeleton_overlay_uses_same_frame_pose() -> None:
    """O overlay deve desenhar as juntas normalizadas do quadro fornecido."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    joints = np.zeros((15, 3), dtype=np.float32)
    joints[:, 0] = 0.5
    joints[:, 1] = 0.5

    overlay = draw_skeleton(frame, joints)

    assert np.array_equal(frame, np.zeros_like(frame))
    assert overlay.shape == frame.shape
    center_patch = overlay[45:56, 95:106]
    assert np.count_nonzero(center_patch) > 0


def test_missing_pose_does_not_draw_false_skeleton() -> None:
    """O sentinela zero da OS-1 nao deve aparecer como pose no canto."""
    frame = np.full((40, 60, 3), 17, dtype=np.uint8)

    overlay = draw_skeleton(
        frame,
        np.zeros((15, 3), dtype=np.float32),
    )

    assert np.array_equal(overlay, frame)


def test_graphical_panel_exposes_clickable_controls() -> None:
    """O painel deve oferecer botoes de classe, navegacao, salvar e cancelar."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    joints = np.zeros((15, 3), dtype=np.float32)
    joints[:, 0] = 0.5
    joints[:, 1] = 0.5

    canvas, buttons, timeline = render_annotation_ui(
        frame,
        joints,
        frame_idx=10,
        frame_count=100,
        label="get_connectors",
        interval_start=4,
        playing=False,
        key_bindings=DEFAULT_KEY_BINDINGS,
        labels=(
            ["no_action"] * 25
            + ["get_connectors"] * 25
            + ["get_screws"] * 25
            + ["ignore"] * 25
        ),
    )

    assert canvas.shape[0] == frame.shape[0]
    assert canvas.shape[1] > frame.shape[1]
    assert {
        "play",
        "mark",
        "four_tubes",
        "undo",
        "save",
        "cancel",
        "class:no_action",
        "class:get_connectors",
        "class:get_screws",
        "class:get_wheels",
        "class:ignore",
    }.issubset(buttons)
    assert timeline[0] >= frame.shape[1]

    timeline_y = (timeline[1] + timeline[3]) // 2
    segment_width = timeline[2] - timeline[0]
    expected_labels = (
        "no_action",
        "get_connectors",
        "get_screws",
        "ignore",
    )
    from datacol.annotate_pkl import LABEL_COLORS

    for segment_index, expected_label in enumerate(expected_labels):
        sample_x = timeline[0] + int(
            (segment_index + 0.5) / len(expected_labels) * segment_width
        )
        assert tuple(int(value) for value in canvas[timeline_y, sample_x]) == (
            LABEL_COLORS[expected_label]
        )


def test_clicks_map_to_buttons_and_timeline() -> None:
    """Hit testing deve distinguir botoes e converter timeline em frame."""
    buttons = {"save": (100, 10, 200, 50)}
    timeline = (100, 60, 200, 80)

    assert action_at_point((150, 30), buttons, timeline, 101) == (
        "save",
        None,
    )
    assert action_at_point((150, 70), buttons, timeline, 101) == (
        "seek",
        51,
    )
    assert action_at_point((20, 20), buttons, timeline, 101) is None


def test_plan_event_round_trip_and_schema(tmp_path: Path) -> None:
    """O evento four_tubes deve ser único, persistente e validado pelo schema."""
    labels = ["no_action"] * 3 + ["ignore"] * 4 + ["no_action"] * 3
    document = empty_plan_events()
    assert toggle_begin_four_tubes(document, 3)
    path = tmp_path / "plan_events.json"

    save_plan_events(path, document, len(labels), labels)
    loaded = load_plan_events(path, len(labels), labels)

    assert loaded == {
        "events": [{"frame_idx": 3, "event": "begin_four_tubes"}]
    }
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "plan_events.schema.json"
    )
    Draft202012Validator(
        json.loads(schema_path.read_text())
    ).validate(loaded)


def test_plan_event_must_start_an_ignore_interval() -> None:
    """A ativação deve marcar exatamente o primeiro quadro do bloco ignore."""
    labels = ["no_action", "ignore", "ignore", "no_action"]

    with pytest.raises(ValueError, match="first frame"):
        validate_plan_events(
            {
                "events": [
                    {"frame_idx": 2, "event": "begin_four_tubes"}
                ]
            },
            len(labels),
            labels,
        )
