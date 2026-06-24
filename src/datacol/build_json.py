"""Consolidação de sessões anotadas no dataset de treinamento."""

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datacol import ANNOTATION_LABELS, INTENTION_LIST
from datacol.annotate_pkl import (
    PLAN_EVENT_BEGIN_FOUR_TUBES,
    annotations_to_labels,
    load_annotations,
    load_plan_events,
)
from datacol.plan_sim import PlanGraph

MODEL_WINDOW = 5
MODEL_JOINTS = 15
MODEL_COORDS = 3
MODEL_CHANNELS = MODEL_JOINTS * MODEL_COORDS
CONTEXT_DIMS = (0, 7, 10)
PLAN_POLICY = "proxy_graph"


def assign_session_splits(
    session_ids: Sequence[str],
    test_session_ids: Sequence[str],
    allow_empty_split: bool = False,
) -> Dict[str, List[str]]:
    """Separa sessões inteiras entre treino e teste.

    Args:
        session_ids: Todas as sessões elegíveis, sem repetição.
        test_session_ids: Subconjunto reservado integralmente para teste.

    Returns:
        Mapa ``{"train": [...], "test": [...]}``, sem sessão compartilhada.

    Raises:
        ValueError: Para IDs desconhecidos, repetidos ou split vazio sem
            autorização explícita.
    """
    all_ids = list(session_ids)
    test_ids = list(test_session_ids)
    if not all_ids:
        raise ValueError("at least one eligible session is required")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("session_ids contains duplicates")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("test_session_ids contains duplicates")

    unknown = sorted(set(test_ids).difference(all_ids))
    if unknown:
        raise ValueError(f"unknown test sessions: {unknown}")

    test_set = set(test_ids)
    train = sorted(session_id for session_id in all_ids if session_id not in test_set)
    test = sorted(test_set)
    if not allow_empty_split and (not train or not test):
        raise ValueError("train and test splits must both contain sessions")
    return {"train": train, "test": test}


def build_windows(
    session_dir: Path,
    context_dim: int = 7,
    window_size: int = MODEL_WINDOW,
) -> List[Dict[str, Any]]:
    """Gera janelas contidas em uma única anotação.

    Args:
        session_dir: Sessão com ``meta.json``, esqueleto e anotações válidas.
        context_dim: Dimensão 0, 7 ou 10 solicitada ao ``PlanGraph``.
        window_size: Número de quadros; o contrato principal usa 5.

    Returns:
        Janelas com pose ``[window_size, 45]``, classe numérica, intenção,
        índices inicial/final e contexto vigente no quadro inicial.
        Intervalos ``ignore`` e janelas que cruzam rótulos são excluídos.

    Raises:
        FileNotFoundError: Se faltar um artefato obrigatório.
        ValueError: Se os artefatos violarem shapes, índices ou metadados.
    """
    session_path = Path(session_dir)
    if context_dim not in CONTEXT_DIMS:
        raise ValueError(f"context_dim must be one of {CONTEXT_DIMS}")
    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size <= 0
    ):
        raise ValueError("window_size must be a positive integer")

    meta = _load_meta(session_path)
    skeleton = _load_skeleton(session_path / "skeleton.pkl")
    annotations = load_annotations(
        session_path / "annotations.json",
        len(skeleton),
    )
    labels = annotations_to_labels(annotations, len(skeleton))
    plan_events = load_plan_events(
        session_path / "plan_events.json",
        len(skeleton),
        labels,
    )
    contexts = _build_context_timeline(labels, context_dim, plan_events)

    windows: List[Dict[str, Any]] = []
    for start in range(0, len(skeleton) - window_size + 1):
        end = start + window_size - 1
        label = labels[start]
        if label == "ignore":
            continue
        if any(candidate != label for candidate in labels[start : end + 1]):
            continue

        pose = skeleton[start : end + 1].reshape(window_size, MODEL_CHANNELS)
        windows.append(
            {
                "session_id": meta["session_id"],
                "frame_idx": start,
                "end_frame_idx": end,
                "intention": label,
                "label": INTENTION_LIST[label],
                "pose": pose.tolist(),
                "context": list(contexts[start]),
            }
        )
    return windows


def build_dataset(
    sessions_root: Path,
    output_json: Path,
    report_path: Path,
    test_session_ids: Sequence[str],
    context_dim: int = 7,
    window_size: int = MODEL_WINDOW,
    allow_empty_split: bool = False,
) -> Tuple[Path, Path]:
    """Consolida sessões no JSON compatível com ``Dataset.py``.

    A hierarquia legada ``split -> intention -> session -> {start, end}`` é
    preservada. Cada sessão também recebe ``windows``; o loader legado ignora
    esse campo, enquanto o treinamento com contexto pode consumi-lo.
    """
    root = Path(sessions_root)
    output = Path(output_json)
    report = Path(report_path)
    if not root.is_dir():
        raise FileNotFoundError(f"sessions root not found: {root}")
    if context_dim not in CONTEXT_DIMS:
        raise ValueError(f"context_dim must be one of {CONTEXT_DIMS}")

    session_dirs = _discover_annotated_sessions(root)
    splits = assign_session_splits(
        [session_dir.name for session_dir in session_dirs],
        test_session_ids,
        allow_empty_split=allow_empty_split,
    )
    split_by_session = {
        session_id: split
        for split, session_ids in splits.items()
        for session_id in session_ids
    }

    dataset: Dict[str, Any] = {
        "_meta": {
            "format_version": 1,
            "window_size": window_size,
            "joints": MODEL_JOINTS,
            "coordinates": MODEL_COORDS,
            "channels": MODEL_CHANNELS,
            "context_dim": context_dim,
            "plan_policy": PLAN_POLICY,
            "intention_list": dict(INTENTION_LIST),
            "splits": splits,
        },
        "train": _empty_split(),
        "test": _empty_split(),
    }
    counts = {
        split: Counter({label: 0 for label in INTENTION_LIST})
        for split in ("train", "test")
    }

    for session_dir in session_dirs:
        session_id = session_dir.name
        split = split_by_session[session_id]
        skeleton = _load_skeleton(session_dir / "skeleton.pkl")
        annotations = load_annotations(
            session_dir / "annotations.json",
            len(skeleton),
        )
        windows = build_windows(session_dir, context_dim, window_size)
        windows_by_label = {
            label: [
                window
                for window in windows
                if window["intention"] == label
            ]
            for label in INTENTION_LIST
        }

        for label in INTENTION_LIST:
            starts = annotations[label]["start"]
            ends = annotations[label]["end"]
            if not starts and not windows_by_label[label]:
                continue
            dataset[split][label][session_id] = {
                # Dataset.py subtrai 1 dos dois limites e trata end como
                # exclusivo. A conversão abaixo preserva [start, end]
                # zero-based e inclusivo da anotação.
                "start": [start + 1 for start in starts],
                "end": [end + 2 for end in ends],
                "windows": windows_by_label[label],
            }
            counts[split][label] += len(windows_by_label[label])

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report.write_text(
        _render_class_report(counts, splits, context_dim, window_size),
        encoding="utf-8",
    )
    return output, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Executa a interface de linha de comando da consolidação."""
    parser = argparse.ArgumentParser(
        description="Consolida sessões anotadas no dataset HRC.",
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path("sessions"),
        help="diretório que contém as sessões",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/v1/dataset.json"),
        help="JSON consolidado de saída",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("datasets/v1/report_classes.md"),
        help="relatório Markdown de classes",
    )
    parser.add_argument(
        "--test-session",
        action="append",
        required=True,
        dest="test_sessions",
        help="ID de sessão reservado para teste; pode ser repetido",
    )
    parser.add_argument(
        "--allow-empty-split",
        action="store_true",
        help=(
            "permite treino ou teste vazio para ensaios técnicos; "
            "não usar na geração do dataset experimental"
        ),
    )
    parser.add_argument(
        "--context-dim",
        type=int,
        choices=CONTEXT_DIMS,
        default=7,
    )
    parser.add_argument("--window-size", type=int, default=MODEL_WINDOW)
    args = parser.parse_args(argv)

    output, report = build_dataset(
        sessions_root=args.sessions_root,
        output_json=args.output,
        report_path=args.report,
        test_session_ids=args.test_sessions,
        context_dim=args.context_dim,
        window_size=args.window_size,
        allow_empty_split=args.allow_empty_split,
    )
    print(f"dataset: {output}")
    print(f"report: {report}")
    return 0


def _load_meta(session_dir: Path) -> Dict[str, Any]:
    meta_path = session_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing session metadata: {meta_path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {meta_path}: {error}") from error
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path} must contain a JSON object")
    if meta.get("session_id") != session_dir.name:
        raise ValueError(
            f"meta session_id {meta.get('session_id')!r} does not match "
            f"directory {session_dir.name!r}"
        )
    return meta


def _load_skeleton(path: Path) -> np.ndarray:
    try:
        with path.open("rb") as skeleton_file:
            skeleton = np.asarray(pickle.load(skeleton_file), dtype=np.float32)
    except FileNotFoundError:
        raise FileNotFoundError(f"missing session skeleton: {path}") from None
    if skeleton.ndim != 3 or skeleton.shape[1:] != (
        MODEL_JOINTS,
        MODEL_COORDS,
    ):
        raise ValueError(
            f"skeleton.pkl must have shape (N, 15, 3); received "
            f"{skeleton.shape}"
        )
    if len(skeleton) == 0:
        raise ValueError("skeleton.pkl cannot be empty")
    if not np.isfinite(skeleton).all():
        raise ValueError("skeleton.pkl contains non-finite values")
    return skeleton


def _build_context_timeline(
    labels: Sequence[str],
    context_dim: int,
    plan_events: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[List[float]]:
    """Calcula o contexto antes da conclusão da intenção de cada intervalo."""
    return [
        frame_state["context"]
        for frame_state in build_plan_timeline(
            labels,
            context_dim,
            plan_events,
        )
    ]


def build_plan_timeline(
    labels: Sequence[str],
    context_dim: int = 7,
    plan_events: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Reconstrói contexto e estado do PlanGraph para cada quadro.

    O estado armazenado em cada posição é o vigente antes da conclusão do
    intervalo anotado naquele quadro, exatamente como o contexto emitido no
    dataset.
    """
    if context_dim not in CONTEXT_DIMS:
        raise ValueError(f"context_dim must be one of {CONTEXT_DIMS}")

    plan = PlanGraph(policy=PLAN_POLICY)
    timeline: List[Dict[str, Any]] = []
    event_map = {
        event["frame_idx"]: event["event"]
        for event in (plan_events or {"events": []})["events"]
    }

    for frame_idx, label in enumerate(labels):
        event = event_map.get(frame_idx)
        if event is not None:
            _apply_plan_event(plan, event)

        context = [] if context_dim == 0 else plan.to_context_vector(context_dim)
        timeline.append(
            {
                "frame_idx": frame_idx,
                "label": label,
                "context": list(context),
                "snapshot": plan.snapshot(),
                "event": event,
            }
        )

        interval_ends = (
            frame_idx == len(labels) - 1
            or labels[frame_idx + 1] != label
        )
        if interval_ends and label in INTENTION_LIST and label != "no_action":
            plan.step(label)

    return timeline


def _apply_plan_event(plan: PlanGraph, event: str) -> None:
    """Aplica um evento externo ao classificador no replay offline."""
    if event != PLAN_EVENT_BEGIN_FOUR_TUBES:
        raise ValueError(f"unknown plan event: {event!r}")
    plan.begin_four_tubes_stage()
    for _ in range(4):
        action = plan.apply_command("short")
        if action is None:
            raise RuntimeError("could not deliver the four short tubes")
        plan.apply_action(action[0])


def _discover_annotated_sessions(root: Path) -> List[Path]:
    sessions = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "meta.json").is_file()
        and (path / "skeleton.pkl").is_file()
        and (path / "annotations.json").is_file()
    )
    if not sessions:
        raise ValueError(f"no annotated sessions found in {root}")
    for session in sessions:
        _load_meta(session)
    return sessions


def _empty_split() -> Dict[str, Dict[str, Any]]:
    return {label: {} for label in INTENTION_LIST}


def _render_class_report(
    counts: Dict[str, Counter],
    splits: Dict[str, List[str]],
    context_dim: int,
    window_size: int,
) -> str:
    lines = [
        "# Relatorio de classes",
        "",
        f"- Janela: `{window_size} x {MODEL_CHANNELS}`",
        f"- Contexto: `{context_dim}D`",
        f"- Politica do plano: `{PLAN_POLICY}`",
        f"- Sessoes de treino: {', '.join(splits['train'])}",
        f"- Sessoes de teste: {', '.join(splits['test'])}",
        "",
        "| Split | Classe | ID | Janelas |",
        "|---|---|---:|---:|",
    ]
    for split in ("train", "test"):
        for label, class_id in INTENTION_LIST.items():
            lines.append(
                f"| {split} | {label} | {class_id} | {counts[split][label]} |"
            )
        lines.append(
            f"| {split} | **total** | - | **{sum(counts[split].values())}** |"
        )
    lines.extend(
        [
            "",
            f"**Total geral:** {sum(sum(value.values()) for value in counts.values())}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
