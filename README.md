# HRC Data Collection

Repositório autônomo para gravação, anotação e consolidação do dataset de
predição de intenção humano-robô usado no projeto de Iniciação Científica
(PIBIC/CNPq, IME/LIARC). Grava sessões com webcam comum e MediaPipe Pose,
anota os intervalos de cada gesto, simula o progresso da montagem
(`PlanGraph`) e consolida tudo em um dataset único, consumido pelo
[`hrc-finetune`](https://github.com/Kalile24/IC_Kalile_Finetune) para
treinar o preditor de intenção.

Este projeto **não depende de ROS, OAK-D ou PyTorch** — é um pacote Python
isolado (`src/datacol/`), pensado para rodar em qualquer notebook com
webcam.

## Contexto do projeto

O preditor de intenção original ([Yu et al., arXiv:2411.15711](https://arxiv.org/abs/2411.15711))
foi treinado com uma câmera de profundidade (OAK-D Lite). Este repositório
recria o pipeline de coleta usando webcam comum + MediaPipe (pose estimada
por software, sem sensor dedicado), e adiciona um **vetor de contexto de
tarefa** (progresso da montagem) que não existia na coleta original — a
ideia central da IC é medir se esse vetor melhora a predição de intenção.

## Pipeline, em ordem

```
capture_session   →   annotate_pkl   →   context_replay   →   build_json
   (grava)           (rotula)          (audita contexto)    (consolida)
```

1. **`capture_session.py`** grava uma sessão: vídeo, pose (MediaPipe, 15
   juntas do tronco superior) e metadados, tudo sincronizado por
   `frame_idx`.
2. **`annotate_pkl.py`** reproduz a sessão gravada, sobrepõe o esqueleto e
   permite marcar manualmente os intervalos de cada classe
   (`no_action`, `get_connectors`, `get_screws`, `get_wheels`, `ignore`).
3. **`context_replay.py`** reproduz vídeo, anotação e estado do
   `PlanGraph` lado a lado, para auditar visualmente o vetor de contexto
   antes de consolidar.
4. **`build_json.py`** junta todas as sessões anotadas em um único JSON,
   com janelas de pose `[5, 45]` e o vetor de contexto (7D ou 10D) já
   calculado por janela, separando treino e teste por sessão inteira.

## Contratos principais

- Classes de intenção: `no_action=0`, `get_connectors=1`, `get_screws=2`,
  `get_wheels=3` (idêntico ao repositório original).
- `ignore` é um rótulo exclusivo da anotação — nunca recebe ID de classe
  nem gera janelas de treinamento.
- `frame_idx` é zero-based e identifica o mesmo quadro em `frames.jsonl`,
  `skeleton.pkl` e `video.mp4`.
- Intervalos em `annotations.json` usam limites inclusivos.
- Cada pose processada tem formato `[15, 3]`, na ordem definida em
  `datacol.joints15`.
- A janela de entrada do modelo tem formato `[B, 5, 45]`.
- O vetor de contexto é produzido exclusivamente por
  `datacol.plan_sim.PlanGraph` (política `proxy_graph`), atualizado apenas
  no evento de confirmação de uma ação, nunca por frame bruto.
- Eventos externos como `begin_four_tubes` ficam em `plan_events.json`,
  separados das classes de intenção.

## Estrutura

```text
src/datacol/         os quatro scripts do pipeline e o mapeamento das 15 juntas
schemas/             JSON Schemas dos artefatos de coleta e anotação
tests/                critérios de aceite de cada etapa (OS-1 a OS-4)
sessions/             sessões gravadas (annotations/frames/meta/plan_events/
                      skeleton versionados; video.mp4 fica só local)
datasets/v1/          datasets consolidados (dataset_dim{0,7,10}.json)
docs/                 manual de cada módulo, um por script
```

## Documentação

| Arquivo | Conteúdo |
|---|---|
| [`docs/capture_session.md`](docs/capture_session.md) | Como gravar uma sessão, opções de câmera, validação. |
| [`docs/annotate_pkl.md`](docs/annotate_pkl.md) | Interface e atalhos do anotador, formato de `annotations.json`. |
| [`docs/context_replay.md`](docs/context_replay.md) | Como auditar o vetor de contexto antes de consolidar. |
| [`docs/plan_sim.md`](docs/plan_sim.md) | `PlanGraph`: políticas, vetores 7D/10D, separação decisão/conclusão. |
| [`docs/build_json.md`](docs/build_json.md) | Formato do dataset consolidado, regras de split, compatibilidade com o loader legado. |
| [`docs/mapeamento_os1-4_vs_original.md`](docs/mapeamento_os1-4_vs_original.md) | Comparação módulo a módulo com o repositório original do artigo. |
| [`docs/planejamento_finetune_os5-7.md`](docs/planejamento_finetune_os5-7.md) | Plano de fine-tuning consumido pelo `hrc-finetune` (datasets, protocolo V0/V1/V2). |

## Desenvolvimento

Requer Python 3.9.

```bash
python -m pip install --upgrade setuptools wheel
python -m pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

O bloqueio de plugins evita que plugins pytest de um ambiente ROS global
contaminem este repositório independente. Sem instalar o pacote em modo
editável, os módulos também podem ser executados diretamente do checkout
com `PYTHONPATH=src`.

## Uso rápido

```bash
# Descobrir câmeras disponíveis
python -m datacol.capture_session --list-cameras

# Gravar uma sessão
python -m datacol.capture_session \
  --participant P01 --script-id R01 \
  --camera-model "Intelbras WCI 1080p" --camera-distance-m 2.2

# Anotar a sessão gravada
python -m datacol.annotate_pkl sessions/<ID_DA_SESSAO>

# Auditar o contexto (sessões com o evento begin_four_tubes)
python -m datacol.context_replay sessions/<ID_DA_SESSAO> --context-dim 7

# Consolidar o dataset, reservando sessões inteiras para teste
python -m datacol.build_json \
  --sessions-root sessions \
  --output datasets/v1/dataset.json \
  --report datasets/v1/report_classes.md \
  --test-session <ID_SESSAO_TESTE_1> \
  --test-session <ID_SESSAO_TESTE_2> \
  --context-dim 7 --window-size 5
```

Guia completo de cada comando, com todas as flags e exemplos: ver a tabela
de documentação acima. O guia operacional de ponta a ponta (gravação →
consolidação → treino → resultados) está em
[`hrc-finetune/reports/GUIA_FINAL.md`](https://github.com/Kalile24/IC_Kalile_Finetune/blob/main/reports/GUIA_FINAL.md).

## Estado atual

8 sessões gravadas (`S01_20260712` a `S08_20260712`, 2 participantes,
roteiros `R01`–`R06`), consolidadas em 3 datasets (`context_dim` 0, 7 e
10), com split de teste fixado em `S05_20260712` + `S02_20260712`. Esses
datasets são consumidos pelo `hrc-finetune` para treinar e avaliar as
variantes V0/V1/V2 do preditor de intenção.
