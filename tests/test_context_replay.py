"""Testes da auditoria grafica do contexto do PlanGraph."""

import numpy as np

from datacol.build_json import build_plan_timeline
from datacol.context_replay import (
    action_at_point,
    draw_proxy_diagram,
    main,
    render_context_ui,
)


def test_plan_timeline_exposes_context_and_snapshot_before_transition() -> None:
    """Cada quadro deve expor o mesmo estado anterior usado pela OS-4."""
    labels = ["get_connectors"] * 5 + ["no_action"] * 5

    timeline = build_plan_timeline(labels, context_dim=7)

    assert timeline[0]["context"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert timeline[4]["snapshot"]["tube_count"] == {"short": 0, "long": 0}
    assert timeline[5]["context"] == [
        0.0,
        1.0,
        0.0,
        0.0,
        0.125,
        0.0,
        0.0,
    ]
    assert timeline[5]["snapshot"]["tube_count"] == {"short": 1, "long": 0}


def test_proxy_diagram_changes_when_assembly_progresses() -> None:
    """O quadro da proxy deve representar visualmente o estado recebido."""
    empty = {
        "stage_record": {"bottom": [], "four_tubes": [], "top": []},
        "screw_count": {"bottom": 0, "four_tubes": 0, "top": 0},
        "wheels_count": 0,
        "stage": None,
    }
    complete = {
        "stage_record": {
            "bottom": ["a"] * 4,
            "four_tubes": ["a"] * 4,
            "top": ["a"] * 4,
        },
        "screw_count": {"bottom": 4, "four_tubes": 4, "top": 4},
        "wheels_count": 4,
        "stage": None,
    }
    empty_canvas = np.zeros((400, 500, 3), dtype=np.uint8)
    complete_canvas = empty_canvas.copy()

    draw_proxy_diagram(empty_canvas, (10, 10, 490, 390), empty)
    draw_proxy_diagram(complete_canvas, (10, 10, 490, 390), complete)

    assert not np.array_equal(empty_canvas, complete_canvas)
    pixel_difference = np.abs(
        complete_canvas.astype(np.int16) - empty_canvas.astype(np.int16)
    )
    assert np.count_nonzero(pixel_difference) > 100


def test_context_ui_combines_video_action_vector_and_timeline() -> None:
    """A tela deve combinar replay, classe, contexto e linha do tempo."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    joints = np.zeros((15, 3), dtype=np.float32)
    labels = ["no_action"] * 5 + ["get_connectors"] * 5
    frame_state = build_plan_timeline(labels, context_dim=7)[5]

    canvas, buttons, timeline_rect = render_context_ui(
        frame,
        joints,
        frame_idx=5,
        frame_count=10,
        label="get_connectors",
        labels=labels,
        frame_state=frame_state,
    )

    assert canvas.shape == (720, 1800, 3)
    assert timeline_rect[0] >= frame.shape[1]
    assert np.count_nonzero(canvas[:, frame.shape[1] :]) > 0
    assert {
        "back10",
        "back1",
        "play",
        "forward1",
        "forward10",
        "save",
    } == set(buttons)


def test_context_replay_clicks_control_navigation_and_timeline() -> None:
    """Botoes e timeline devem produzir comandos de navegacao inequívocos."""
    buttons = {
        "back1": (100, 10, 150, 40),
        "save": (160, 10, 230, 40),
    }
    timeline = (100, 50, 300, 70)

    assert action_at_point((125, 25), buttons, timeline, 101) == (
        "back1",
        None,
    )
    assert action_at_point((190, 25), buttons, timeline, 101) == (
        "save",
        None,
    )
    assert action_at_point((200, 60), buttons, timeline, 101) == (
        "seek",
        50,
    )


def test_cli_paused_flag_disables_autoplay(monkeypatch, tmp_path) -> None:
    """A CLI deve reproduzir por padrao e aceitar abertura pausada."""
    received = []

    def fake_replay(
        session_dir,
        context_dim,
        start_frame,
        autoplay=True,
        export_path=None,
    ):
        received.append(
            (session_dir, context_dim, start_frame, autoplay, export_path)
        )

    monkeypatch.setattr("datacol.context_replay.replay_context", fake_replay)

    assert main([str(tmp_path), "--context-dim", "10"]) == 0
    assert main([str(tmp_path), "--paused", "--start-frame", "12"]) == 0
    assert received == [
        (tmp_path, 10, 0, True, None),
        (tmp_path, 7, 12, False, None),
    ]


def test_cli_export_only_writes_requested_video(monkeypatch, tmp_path) -> None:
    """O modo não interativo deve exportar no caminho solicitado."""
    output = tmp_path / "audit.mp4"
    received = []

    def fake_export(session_dir, output_path, context_dim):
        received.append((session_dir, output_path, context_dim))
        return output_path

    monkeypatch.setattr(
        "datacol.context_replay.export_context_video",
        fake_export,
    )

    assert main(
        [
            str(tmp_path),
            "--context-dim",
            "10",
            "--output-video",
            str(output),
            "--export-only",
        ]
    ) == 0
    assert received == [(tmp_path, output, 10)]
