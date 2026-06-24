"""Critérios de aceite da consolidação do dataset."""

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from datacol.annotate_pkl import labels_to_annotations
from datacol.build_json import (
    assign_session_splits,
    build_dataset,
    build_windows,
    main,
)


def _write_session(root: Path, session_id: str, labels: list) -> Path:
    session = root / session_id
    session.mkdir(parents=True)
    skeleton = np.arange(
        len(labels) * 15 * 3,
        dtype=np.float32,
    ).reshape(len(labels), 15, 3)
    with (session / "skeleton.pkl").open("wb") as skeleton_file:
        pickle.dump(skeleton, skeleton_file)
    (session / "annotations.json").write_text(
        json.dumps(labels_to_annotations(labels)),
        encoding="utf-8",
    )
    (session / "meta.json").write_text(
        json.dumps({"session_id": session_id}),
        encoding="utf-8",
    )
    return session


def _write_plan_event(session: Path, frame_idx: int) -> None:
    (session / "plan_events.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "frame_idx": frame_idx,
                        "event": "begin_four_tubes",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_model_windows_have_shape_five_by_forty_five(tmp_path: Path) -> None:
    """Cada entrada representa cinco quadros de 15 juntas XYZ."""
    session = _write_session(
        tmp_path,
        "S01_20260613",
        ["get_connectors"] * 7,
    )

    windows = build_windows(session)

    assert len(windows) == 3
    assert np.asarray(windows[0]["pose"]).shape == (5, 45)
    assert windows[0]["label"] == 1
    assert len(windows[0]["context"]) == 7


def test_ignore_frames_never_generate_windows(tmp_path: Path) -> None:
    """Nenhuma janela pode conter ou usar como alvo um quadro ignore."""
    labels = ["no_action"] * 6 + ["ignore"] * 3 + ["get_screws"] * 6
    session = _write_session(tmp_path, "S01_20260613", labels)

    windows = build_windows(session)

    assert {window["frame_idx"] for window in windows} == {0, 1, 9, 10}
    assert all(
        "ignore" not in labels[
            window["frame_idx"] : window["end_frame_idx"] + 1
        ]
        for window in windows
    )


def test_window_context_matches_initial_frame_state(tmp_path: Path) -> None:
    """O contexto é o estado do PlanGraph no primeiro quadro da janela."""
    labels = (
        ["get_connectors"] * 5
        + ["no_action"] * 5
        + ["get_connectors"] * 5
    )
    session = _write_session(tmp_path, "S01_20260613", labels)

    windows = build_windows(session)
    by_start = {window["frame_idx"]: window for window in windows}

    assert by_start[0]["context"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert by_start[5]["context"] == pytest.approx(
        [0.0, 1.0, 0.0, 0.0, 1.0 / 8.0, 0.0, 0.0]
    )
    assert by_start[10]["context"] == pytest.approx(
        [0.0, 1.0, 0.0, 0.0, 1.0 / 8.0, 0.0, 0.0]
    )


def test_session_split_has_no_leakage() -> None:
    """Uma sessão pertence integralmente a apenas um split."""
    splits = assign_session_splits(
        ["S03_20260613", "S01_20260613", "S02_20260613"],
        ["S02_20260613"],
    )

    assert splits == {
        "train": ["S01_20260613", "S03_20260613"],
        "test": ["S02_20260613"],
    }
    assert set(splits["train"]).isdisjoint(splits["test"])


def test_class_report_counts_all_emitted_windows(tmp_path: Path) -> None:
    """O relatório confere com as janelas emitidas por classe e split."""
    sessions = tmp_path / "sessions"
    _write_session(
        sessions,
        "S01_20260613",
        ["no_action"] * 6 + ["get_connectors"] * 5,
    )
    _write_session(
        sessions,
        "S02_20260613",
        ["get_screws"] * 5 + ["get_wheels"] * 6,
    )
    output = tmp_path / "dataset.json"
    report = tmp_path / "report.md"

    build_dataset(
        sessions,
        output,
        report,
        test_session_ids=["S02_20260613"],
    )

    dataset = json.loads(output.read_text(encoding="utf-8"))
    assert dataset["_meta"]["plan_policy"] == "proxy_graph"
    assert set(dataset["train"]["no_action"]) == {"S01_20260613"}
    assert set(dataset["test"]["get_screws"]) == {"S02_20260613"}
    assert dataset["train"]["get_connectors"]["S01_20260613"]["start"] == [7]
    assert dataset["train"]["get_connectors"]["S01_20260613"]["end"] == [12]

    report_text = report.read_text(encoding="utf-8")
    assert "| train | no_action | 0 | 2 |" in report_text
    assert "| train | get_connectors | 1 | 1 |" in report_text
    assert "| test | get_screws | 2 | 1 |" in report_text
    assert "| test | get_wheels | 3 | 2 |" in report_text
    assert "**Total geral:** 6" in report_text


def test_context_dim_zero_emits_empty_context(tmp_path: Path) -> None:
    """O baseline sem contexto mantém janelas com vetor vazio."""
    session = _write_session(
        tmp_path,
        "S01_20260613",
        ["no_action"] * 5,
    )

    windows = build_windows(session, context_dim=0)

    assert windows[0]["context"] == []


def test_four_tubes_event_updates_context_after_ignore(tmp_path: Path) -> None:
    """A entrega externa de quatro tubos deve aparecer após o bloco ignore."""
    labels = []
    for _ in range(4):
        labels.extend(["get_connectors"] * 5)
        labels.extend(["no_action"] * 5)
    event_frame = len(labels)
    labels.extend(["ignore"] * 5)
    labels.extend(["no_action"] * 5)
    session = _write_session(tmp_path, "S01_20260613", labels)
    _write_plan_event(session, event_frame)

    windows = build_windows(session)
    first_after_event = next(
        window for window in windows if window["frame_idx"] == event_frame + 5
    )

    assert first_after_event["context"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("four_tubes_first", [False, True])
def test_complete_branch_paths_reach_full_context(
    tmp_path: Path,
    four_tubes_first: bool,
) -> None:
    """A OS-4 reproduz os dois caminhos completos do grafo da proxy."""
    labels = ["no_action"] * 5

    def add_intentions(label: str, count: int) -> None:
        for _ in range(count):
            labels.extend([label] * 5)
            labels.extend(["no_action"] * 5)

    add_intentions("get_connectors", 4)
    add_intentions("get_screws", 4)

    event_frame = -1
    branches = ("four_tubes", "top") if four_tubes_first else ("top", "four_tubes")
    for branch in branches:
        if branch == "top":
            add_intentions("get_connectors", 4)
        else:
            event_frame = len(labels)
            labels.extend(["ignore"] * 5)
            labels.extend(["no_action"] * 5)
        add_intentions("get_screws", 4)

    add_intentions("get_wheels", 4)
    session = _write_session(tmp_path, "S01_20260613", labels)
    _write_plan_event(session, event_frame)

    windows = build_windows(session)

    assert windows[-1]["context"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]


def test_invalid_or_empty_splits_are_rejected() -> None:
    """IDs desconhecidos, duplicados e splits vazios falham."""
    with pytest.raises(ValueError, match="unknown test"):
        assign_session_splits(["S01", "S02"], ["S03"])
    with pytest.raises(ValueError, match="duplicates"):
        assign_session_splits(["S01", "S01"], ["S01"])
    with pytest.raises(ValueError, match="both contain"):
        assign_session_splits(["S01"], ["S01"])


def test_single_session_build_can_explicitly_allow_empty_train_split(
    tmp_path: Path,
) -> None:
    """Um ensaio técnico pode gerar dataset com apenas uma sessão de teste."""
    sessions = tmp_path / "sessions"
    _write_session(sessions, "S01_20260613", ["no_action"] * 5)
    output = tmp_path / "dataset.json"
    report = tmp_path / "report.md"

    build_dataset(
        sessions,
        output,
        report,
        test_session_ids=["S01_20260613"],
        allow_empty_split=True,
    )

    dataset = json.loads(output.read_text(encoding="utf-8"))
    assert dataset["_meta"]["splits"] == {
        "train": [],
        "test": ["S01_20260613"],
    }
    assert dataset["train"] == {
        label: {} for label in ("no_action", "get_connectors", "get_screws", "get_wheels")
    }
    assert "S01_20260613" in dataset["test"]["no_action"]


def test_cli_writes_dataset_and_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A CLI grava e anuncia os dois artefatos solicitados."""
    sessions = tmp_path / "sessions"
    _write_session(sessions, "S01_20260613", ["no_action"] * 5)
    _write_session(sessions, "S02_20260613", ["get_connectors"] * 5)
    output = tmp_path / "out" / "dataset.json"
    report = tmp_path / "out" / "report.md"

    exit_code = main(
        [
            "--sessions-root",
            str(sessions),
            "--output",
            str(output),
            "--report",
            str(report),
            "--test-session",
            "S02_20260613",
            "--context-dim",
            "10",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert output.is_file()
    assert report.is_file()
    assert f"dataset: {output}" in captured
    assert f"report: {report}" in captured
