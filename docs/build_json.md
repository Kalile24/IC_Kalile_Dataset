# Guia do `build_json.py`

O `build_json.py` implementa a OS-4: lê sessões anotadas, gera janelas de pose,
deriva o contexto pelo `PlanGraph`, separa sessões inteiras entre treino e
teste e escreve um relatório de classes.

## Sessões elegíveis

Uma sessão entra na consolidação quando contém:

```text
sessions/S01_20260620/
├── meta.json
├── skeleton.pkl
├── annotations.json
└── plan_events.json
```

Diretórios de capturas ainda não anotadas são ignorados. O `session_id` de
`meta.json` deve ser igual ao nome do diretório.

São necessárias pelo menos duas sessões anotadas: uma para treino e outra para
teste. O roteiro recomenda reservar ao menos duas sessões completas para
teste quando a coleta definitiva estiver disponível.

## Executar

Com a `.venv` ativa:

```bash
python -m datacol.build_json \
  --sessions-root sessions \
  --output datasets/v1/dataset.json \
  --report datasets/v1/report_classes.md \
  --test-session S09_20260620 \
  --test-session S10_20260620 \
  --context-dim 7 \
  --window-size 5
```

Para um ensaio técnico com apenas uma sessão, permita explicitamente o split
de treino vazio:

```bash
python -m datacol.build_json \
  --sessions-root pilot_sessions \
  --output datasets/v1/pilot_dataset.json \
  --report datasets/v1/pilot_report_classes.md \
  --test-session S01_20260613 \
  --context-dim 7 \
  --window-size 5 \
  --allow-empty-split
```

Esse modo serve apenas para verificar janelamento e contexto. O dataset
experimental continua exigindo sessões distintas de treino e teste.

`--test-session` pode ser repetido. Qualquer sessão anotada que não estiver
nessa lista pertence ao treino. IDs inexistentes, repetidos ou um split vazio
causam erro, evitando divisão silenciosa incorreta.

Dimensões de contexto aceitas:

- `0`: baseline sem contexto, com vetor `[]`;
- `7`: configuração principal;
- `10`: ablação desagregada.

## Janelamento

Para cada quadro inicial possível, o programa verifica os cinco quadros da
janela. Uma janela só é emitida quando:

- os cinco quadros pertencem ao mesmo rótulo;
- nenhum quadro é `ignore`;
- o esqueleto possui o shape contratado `(15, 3)`.

Limitação atual: o sentinela zero usado quando o MediaPipe não detecta pose
ainda satisfaz esse shape. A exclusão automática dessas janelas está
registrada como prioridade antes da coleta definitiva.

A pose é achatada por quadro:

```text
[5, 15, 3] -> [5, 45]
```

Cada janela contém:

```json
{
  "session_id": "S01_20260620",
  "frame_idx": 120,
  "end_frame_idx": 124,
  "intention": "get_connectors",
  "label": 1,
  "pose": [[0.0, 0.0, 0.0]],
  "context": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

No arquivo real, `pose` possui cinco linhas com 45 valores cada.

## Contexto temporal

As anotações descrevem o gesto humano que provoca uma ação do controlador.
Por isso, todas as janelas de um intervalo recebem o contexto vigente antes
daquela ação. A transição do `PlanGraph` é aplicada depois do quadro final do
intervalo.

Exemplo:

```text
get_connectors [10, 30]  -> contexto anterior em todas as janelas
no_action      [31, 50]  -> contexto já atualizado
```

Essa ordem impede vazamento do rótulo: uma janela de `get_connectors` não vê
um contador que já inclua o próprio conector que ela deve predizer.

O estágio `four_tubes` é ativado pelo comando externo `short`, sem uma classe
de intenção própria. O anotador registra esse momento em `plan_events.json`.
O evento deve apontar para o primeiro quadro do bloco `ignore` que representa
a entrega dos quatro tubos curtos. A OS-4 aplica a entrega em lote antes das
próximas janelas válidas.

O dataset registra `plan_policy: "proxy_graph"`, que suporta os dois caminhos,
quatro parafusos por estágio e quatro rodas depois dos 12 parafusos.

## JSON gerado

O arquivo preserva a hierarquia esperada pelo `Dataset.py`:

```text
split -> intention -> session -> {start, end, windows}
```

Exemplo abreviado:

```json
{
  "_meta": {
    "window_size": 5,
    "channels": 45,
    "context_dim": 7,
    "plan_policy": "proxy_graph"
  },
  "train": {
    "get_connectors": {
      "S01_20260620": {
        "start": [11],
        "end": [32],
        "windows": []
      }
    }
  },
  "test": {}
}
```

`start` e `end` são convertidos para os cut points usados pelo loader legado.
O `Dataset.py` continua lendo essas duas chaves e ignora `windows`. O novo
treinamento com contexto consome as janelas materializadas.

Nenhuma sessão pode aparecer nos dois splits.

## Relatório

`report_classes.md` registra:

- IDs das sessões em treino e teste;
- dimensão da janela e do contexto;
- número de janelas por classe e split;
- totais por split e total geral.

O relatório conta somente janelas realmente emitidas, depois da exclusão de
`ignore` e das fronteiras entre rótulos.

## Compatibilidade conhecida

O JSON preserva a hierarquia e os cut points esperados pelo `Dataset.py`.
Entretanto, o novo `skeleton.pkl` é um array `(N, 15, 3)`, enquanto o caminho
PKL do loader legado espera objetos com atributo `.landmarks`. A futura OS-5
deve consumir as janelas materializadas ou usar um adaptador explícito.

O pré-processamento numérico legado também ainda não é aplicado às poses
gravadas no JSON. Não use o dataset piloto para treinamento experimental antes
de resolver os itens registrados em
[`session_status_2026-06-13.md`](session_status_2026-06-13.md).

## Testes

Execute apenas a OS-4:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_build_json.py
```

Para validar todo o repositório:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```
