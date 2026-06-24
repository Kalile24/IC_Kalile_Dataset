# HRC Data Collection

Repositorio autonomo para coleta, anotacao e consolidacao de dados de
trajetorias humanas com webcam e MediaPipe Pose. Este projeto nao depende de
ROS, OAK-D ou PyTorch.

## Contratos principais

- Classes: `no_action=0`, `get_connectors=1`, `get_screws=2`,
  `get_wheels=3`.
- `ignore` e um rotulo exclusivo da anotacao e nunca recebe ID de classe nem
  gera janelas de treinamento.
- `frame_idx` e zero-based e identifica o mesmo quadro em `frames.jsonl`,
  `skeleton.pkl` e `video.mp4`.
- Intervalos em `annotations.json` usam limites inclusivos.
- Cada pose processada tem formato `[15, 3]`, na ordem definida em
  `datacol.joints15`.
- A janela de entrada do modelo tem formato `[B, 5, 45]`.
- O contexto da tarefa e produzido exclusivamente por
  `datacol.plan_sim.PlanGraph`.
- Eventos externos como `begin_four_tubes` ficam em `plan_events.json`.

## Estrutura

```text
schemas/       JSON Schemas dos artefatos de coleta e anotacao
src/datacol/   contratos dos quatro scripts e mapping das 15 juntas
tests/         criterios de aceite das OS-1 a OS-4
sessions/      dados brutos locais, ignorados pelo Git
datasets/v1/   datasets consolidados locais, ignorados pelo Git
```

`annotations.json` mantem, para cada rotulo, listas paralelas `start` e `end`.
O `build_json.py` transforma esses intervalos no formato hierarquico consumido
por `Dataset.py`: `split -> intention -> session task -> {start, end}`. Nessa
etapa, os limites zero-based/inclusivos da anotacao sao convertidos para a
convencao de cut points legada usada pelo loader. Campos adicionais de contexto
podem ser acrescentados sem alterar as chaves `start` e `end`.

## Desenvolvimento

Requer Python 3.9.

```bash
python -m pip install --upgrade setuptools wheel
python -m pip install -e ".[dev]"
pytest
```

Sem instalar o pacote em modo editavel, os modulos tambem podem ser executados
diretamente do checkout com `PYTHONPATH=src`.

As OS-1, OS-2, OS-3 e OS-4 estao implementadas e testadas. O estado de
encerramento, as incompatibilidades conhecidas e a ordem de retomada estão em
[`docs/session_status_2026-06-13.md`](docs/session_status_2026-06-13.md).

## Captura

O logger oferece descoberta automatica de camera, sugestao de ID, HUD com
FPS/status da pose e validacao de sessao:

```bash
python -m datacol.capture_session --list-cameras
python -m datacol.capture_session --suggest-session-id
python -m datacol.capture_session \
  --participant P01 \
  --script-id R02 \
  --camera-model "Intelbras WCI 1080p" \
  --camera-distance-m 2.2
```

O funcionamento completo, os artefatos e os procedimentos de recuperacao
estao documentados em [`docs/capture_session.md`](docs/capture_session.md).

## Anotacao

Para reproduzir uma sessao, sobrepor o esqueleto e marcar os intervalos:

```bash
python -m datacol.annotate_pkl sessions/S01_20260620
```

Os controles, classes e invariantes de `annotations.json` estao documentados
em [`docs/annotate_pkl.md`](docs/annotate_pkl.md).

## Simulacao do plano

O `PlanGraph` offline reproduz as regras de transicao do controlador sem ROS,
waypoints ou execucao fisica:

```python
from datacol.plan_sim import PlanGraph

plan = PlanGraph()
context_before_action = plan.to_context_vector(dim=7)
action = plan.step("get_connectors")
```

Os vetores 7D/10D, a separacao entre decisao e conclusao e a equivalencia com
o preset `--stageI_done` estao documentados em
[`docs/plan_sim.md`](docs/plan_sim.md).

Para auditar graficamente vídeo, anotação e contexto:

```bash
python -m datacol.context_replay sessions/S01_20260620
```

Consulte [`docs/context_replay.md`](docs/context_replay.md).

## Consolidacao

Com pelo menos duas sessoes anotadas, gere o dataset reservando sessoes
inteiras para teste:

```bash
python -m datacol.build_json \
  --test-session S09_20260620 \
  --test-session S10_20260620 \
  --context-dim 7
```

O comando grava `datasets/v1/dataset.json` e
`datasets/v1/report_classes.md`. O formato das janelas, a compatibilidade com
`Dataset.py` e as regras do split estao documentados em
[`docs/build_json.md`](docs/build_json.md).

Antes da coleta definitiva, o piloto da proxy real deve seguir
[`docs/pilot_proxy.md`](docs/pilot_proxy.md). O roteiro usa duas sessões para
testar os caminhos `top -> four_tubes` e `four_tubes -> top`, registrando a
ativação externa em `plan_events.json`, sem incorporar a sessão exploratória
usada apenas para desenvolvimento.

O piloto técnico atual não constitui dataset experimental. Antes da coleta em
volume, devem ser resolvidas as decisões de amostragem temporal,
pré-processamento e compatibilidade do loader registradas no documento de
estado.
