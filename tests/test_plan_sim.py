"""Testes de aceite do simulador offline do PlanGraph."""

import pytest

from datacol.plan_sim import PlanGraph


def _complete_bottom_stage(plan: PlanGraph) -> None:
    for _ in range(4):
        action = plan.apply_intention("get_connectors")
        assert action is not None
        plan.apply_action(action[0])


def test_initial_context_vector() -> None:
    """O plano vazio começa no estágio none com todos os contadores zerados."""
    plan = PlanGraph()

    assert plan.to_context_vector(7) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_context_dimensions_and_ranges() -> None:
    """Os contextos 7D/10D têm dimensões e normalização contratadas."""
    plan = PlanGraph(stageI_done=True)

    context7 = plan.to_context_vector(7)
    context10 = plan.to_context_vector(10)

    assert context7 == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 0.5, 4.0 / 12.0, 0.0]
    )
    assert context10 == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 0.0, 0.0, 0.0]
    )
    assert len(context7) == 7
    assert len(context10) == 10
    assert all(0.0 <= value <= 1.0 for value in context7 + context10)


def test_canonical_sequence_matches_stageI_done_preset() -> None:
    """A sequência canônica do estágio I equivale ao preset --stageI_done."""
    replayed = PlanGraph()
    _complete_bottom_stage(replayed)
    for _ in range(4):
        action = replayed.apply_intention("get_screws")
        assert action is not None
        replayed.apply_action(action[0])

    preset = PlanGraph(stageI_done=True)

    assert replayed.tube_count == preset.tube_count
    assert replayed.screw_count == preset.screw_count
    assert replayed.wheels_count == preset.wheels_count
    assert replayed.stage_record == preset.stage_record
    assert replayed.stage_history == preset.stage_history
    assert replayed.stage == preset.stage


def test_transitions_are_deterministic() -> None:
    """Duas execuções da mesma sequência produzem snapshots idênticos."""
    left = PlanGraph()
    right = PlanGraph()

    for intention in ["get_connectors"] * 4 + ["get_screws"] * 4:
        left.step(intention)
        right.step(intention)

    assert left.snapshot() == right.snapshot()


def test_decision_and_action_completion_are_separate() -> None:
    """Tubos só entram no estado após a ação decidida ser confirmada."""
    plan = PlanGraph()

    action = plan.apply_intention("get_connectors")

    assert action == ("get_short_tubes", 0)
    assert plan.stage == "bottom"
    assert plan.tube_count == {"short": 0, "long": 0}
    assert plan.stage_record["bottom"] == []

    plan.apply_action(action[0])

    assert plan.tube_count == {"short": 1, "long": 0}
    assert plan.stage_record["bottom"] == ["get_short_tubes"]


def test_connector_policy_alternates_and_completes_stage() -> None:
    """Bottom recebe dois tubos curtos e dois longos em ordem alternada."""
    plan = PlanGraph()
    actions = []

    for _ in range(4):
        action = plan.step("get_connectors")
        assert action is not None
        actions.append(action[0])

    assert actions == [
        "get_short_tubes",
        "get_long_tubes",
        "get_short_tubes",
        "get_long_tubes",
    ]
    assert plan.stage is None
    assert plan.stage_history == ["bottom"]


def test_four_tubes_stage_is_explicit_and_uses_manual_commands() -> None:
    """O estágio four_tubes reproduz a ativação externa ao decisor legado."""
    plan = PlanGraph()
    _complete_bottom_stage(plan)
    plan.begin_four_tubes_stage()

    for _ in range(4):
        action = plan.apply_command("short")
        assert action is not None
        plan.apply_action(action[0])

    assert plan.stage is None
    assert plan.stage_history == ["bottom", "four_tubes"]
    assert plan.stage_record["four_tubes"] == ["get_short_tubes"] * 4


def test_stage_graph_accepts_both_branch_orders() -> None:
    """Top e four_tubes podem ser concluídos em qualquer ordem após bottom."""
    top_first = PlanGraph()
    _complete_bottom_stage(top_first)
    for _ in range(4):
        assert top_first.step("get_connectors") is not None
    top_first.begin_four_tubes_stage()
    for _ in range(4):
        action = top_first.apply_command("short")
        assert action is not None
        top_first.apply_action(action[0])

    four_first = PlanGraph()
    _complete_bottom_stage(four_first)
    four_first.begin_four_tubes_stage()
    for _ in range(4):
        action = four_first.apply_command("short")
        assert action is not None
        four_first.apply_action(action[0])
    for _ in range(4):
        assert four_first.step("get_connectors") is not None

    assert top_first.stage_history == ["bottom", "top", "four_tubes"]
    assert four_first.stage_history == ["bottom", "four_tubes", "top"]


@pytest.mark.parametrize(
    "branch_order",
    [
        ("top", "four_tubes"),
        ("four_tubes", "top"),
    ],
)
def test_proxy_graph_completes_both_paths(branch_order: tuple) -> None:
    """O grafo da proxy completa quatro parafusos por ramo e quatro rodas."""
    plan = PlanGraph(policy="proxy_graph")
    _complete_bottom_stage(plan)
    for _ in range(4):
        assert plan.step("get_screws") is not None

    for branch in branch_order:
        if branch == "top":
            for _ in range(4):
                assert plan.step("get_connectors") is not None
        else:
            plan.begin_four_tubes_stage()
            for _ in range(4):
                action = plan.apply_command("short")
                assert action is not None
                plan.apply_action(action[0])
        for _ in range(4):
            assert plan.step("get_screws") is not None

    for _ in range(4):
        assert plan.step("get_wheels") is not None

    assert plan.screw_count == {"bottom": 4, "four_tubes": 4, "top": 4}
    assert plan.wheels_count == 4
    assert plan.to_context_vector(7) == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]


def test_manual_tube_command_outside_stage_updates_only_global_state() -> None:
    """Comando manual sem estágio replica o histórico global do receptor."""
    plan = PlanGraph()

    action = plan.apply_command("long")
    assert action == ("get_long_tubes", 0)
    plan.apply_action(action[0])

    assert plan.tube_count["long"] == 1
    assert plan.action_history == ["get_long_tubes"]
    assert all(not record for record in plan.stage_record.values())


def test_no_action_does_not_change_state() -> None:
    """A classe no_action preserva integralmente o estado corrente."""
    plan = PlanGraph()
    before = plan.snapshot()

    assert plan.apply_intention("no_action") is None
    assert plan.snapshot() == before


def test_invalid_contract_values_are_rejected() -> None:
    """Dimensões, intenções e ações fora do contrato falham explicitamente."""
    plan = PlanGraph()

    with pytest.raises(ValueError, match="context dimension"):
        plan.to_context_vector(8)
    with pytest.raises(ValueError, match="unknown intention"):
        plan.apply_intention("unknown")
    with pytest.raises(ValueError, match="unknown action"):
        plan.apply_action("unknown")
