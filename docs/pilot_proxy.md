# Tutorial do teste completo da proxy

Este teste valida, em ordem:

```text
captura -> integridade -> anotação -> PlanGraph -> janelas -> split -> relatório
```

Serão gravadas duas sessões:

```text
R01: bottom -> top -> four_tubes -> wheels
R02: bottom -> four_tubes -> top -> wheels
```

Use R01 como treino e R02 como teste.

## 1. Preparar o ambiente

No terminal:

```bash
cd /home/marcos-kalile/hrc-data-collection
source .venv/bin/activate

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m datacol.capture_session --list-cameras
```

Antes de coletar, a suíte deve passar. O programa tenta selecionar
automaticamente a primeira webcam disponível. Se a câmera correta não for
selecionada, acrescente ao comando de captura o índice numérico exibido, por
exemplo `--camera-index 2`.

Os IDs serão guardados nas variáveis `ID_R01` e `ID_R02`. Execute todos os
comandos deste tutorial no mesmo terminal para preservar essas variáveis.

## 2. Preparar a bancada

Separe, para cada montagem:

- 8 conectores;
- 12 parafusos;
- 4 rodas;
- 8 tubos curtos;
- 4 tubos longos.

Organize:

- conectores à direita;
- parafusos à esquerda, próximos;
- rodas à esquerda, atrás dos parafusos;
- gabarito ao centro;
- rack de tubos na lateral;
- marca de repouso na borda da bancada.

Mantenha 40 a 50 cm entre os centros das zonas. Não altere câmera,
iluminação, cadeira ou gabarito entre as sessões.

Distribuição obrigatória:

```text
bottom:     curto, longo, curto, longo
top:        curto, longo, curto, longo
four_tubes: curto, curto, curto, curto
```

## 3. Regra de execução

Para toda intenção:

1. Comece com a mão parada na marca de repouso.
2. Mova a mão até a zona do componente.
3. Pegue e deposite o componente no gabarito.
4. Volte à marca de repouso.
5. Espere 2 segundos antes do próximo acontecimento.

Na anotação:

- saída do repouso até o depósito: classe da intenção;
- retorno e espera: `no_action`;
- qualquer manipulação de tubos: `ignore`.

Não faça duas intenções consecutivas sem retornar ao repouso. Cada intenção
precisa virar um intervalo separado em `annotations.json`.

## 4. Gravar R01

Inicie:

```bash
ID_R01=$(python -m datacol.capture_session \
  --suggest-session-id \
  --output-root pilot_sessions)
printf 'Sessão R01: %s\n' "$ID_R01"

python -m datacol.capture_session "$ID_R01" \
  --participant P01 \
  --script-id R01 \
  --camera-model "Intelbras WCI 1080p" \
  --camera-distance-m 2.2 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --output-root pilot_sessions
```

Espere 3 segundos no repouso e execute exatamente:

| Nº | Acontecimento | Rótulo posterior |
|---:|---|---|
| 1 | Conector 1 para `bottom` | `get_connectors` |
| 2 | Tubo curto para `bottom` | `ignore` |
| 3 | Conector 2 para `bottom` | `get_connectors` |
| 4 | Tubo longo para `bottom` | `ignore` |
| 5 | Conector 3 para `bottom` | `get_connectors` |
| 6 | Tubo curto para `bottom` | `ignore` |
| 7 | Conector 4 para `bottom` | `get_connectors` |
| 8 | Tubo longo para `bottom` | `ignore` |
| 9 | Parafuso 1 para `bottom` | `get_screws` |
| 10 | Parafuso 2 para `bottom` | `get_screws` |
| 11 | Parafuso 3 para `bottom` | `get_screws` |
| 12 | Parafuso 4 para `bottom` | `get_screws` |
| 13 | Conector 1 para `top` | `get_connectors` |
| 14 | Tubo curto para `top` | `ignore` |
| 15 | Conector 2 para `top` | `get_connectors` |
| 16 | Tubo longo para `top` | `ignore` |
| 17 | Conector 3 para `top` | `get_connectors` |
| 18 | Tubo curto para `top` | `ignore` |
| 19 | Conector 4 para `top` | `get_connectors` |
| 20 | Tubo longo para `top` | `ignore` |
| 21 | Parafuso 1 para `top` | `get_screws` |
| 22 | Parafuso 2 para `top` | `get_screws` |
| 23 | Parafuso 3 para `top` | `get_screws` |
| 24 | Parafuso 4 para `top` | `get_screws` |
| 25 | Levar os 4 tubos curtos para `four_tubes` | `ignore` único |
| 26 | Parafuso 1 para `four_tubes` | `get_screws` |
| 27 | Parafuso 2 para `four_tubes` | `get_screws` |
| 28 | Parafuso 3 para `four_tubes` | `get_screws` |
| 29 | Parafuso 4 para `four_tubes` | `get_screws` |
| 30 | Roda 1 para o gabarito | `get_wheels` |
| 31 | Roda 2 para o gabarito | `get_wheels` |
| 32 | Roda 3 para o gabarito | `get_wheels` |
| 33 | Roda 4 para o gabarito | `get_wheels` |

Após o nº 33, espere 3 segundos no repouso e pressione `q`.

## 5. Gravar R02

Use outro ID e `--script-id R02`:

```bash
ID_R02=$(python -m datacol.capture_session \
  --suggest-session-id \
  --output-root pilot_sessions)
printf 'Sessão R02: %s\n' "$ID_R02"

python -m datacol.capture_session "$ID_R02" \
  --participant P01 \
  --script-id R02 \
  --camera-model "Intelbras WCI 1080p" \
  --camera-distance-m 2.2 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --output-root pilot_sessions
```

Espere 3 segundos no repouso e execute:

| Nº | Acontecimento | Rótulo posterior |
|---:|---|---|
| 1–12 | Mesmo bloco `bottom` de R01 | mesmos rótulos |
| 13 | Levar os 4 tubos curtos para `four_tubes` | `ignore` único |
| 14–17 | Parafusos 1–4 para `four_tubes` | `get_screws` |
| 18 | Conector 1 para `top` | `get_connectors` |
| 19 | Tubo curto para `top` | `ignore` |
| 20 | Conector 2 para `top` | `get_connectors` |
| 21 | Tubo longo para `top` | `ignore` |
| 22 | Conector 3 para `top` | `get_connectors` |
| 23 | Tubo curto para `top` | `ignore` |
| 24 | Conector 4 para `top` | `get_connectors` |
| 25 | Tubo longo para `top` | `ignore` |
| 26–29 | Parafusos 1–4 para `top` | `get_screws` |
| 30–33 | Rodas 1–4 | `get_wheels` |

Após o nº 33, espere 3 segundos no repouso e pressione `q`.

## 6. Validar as capturas

```bash
python -m datacol.capture_session \
  --validate-session "pilot_sessions/$ID_R01"

python -m datacol.capture_session \
  --validate-session "pilot_sessions/$ID_R02"
```

Cada comando deve retornar `"valid": true`.

Também abra os vídeos e confira:

- corpo e mãos visíveis;
- esqueleto sem defasagem progressiva;
- zonas não encobertas;
- nenhum trecho importante fora do enquadramento.

## 7. Anotar R01 e R02

```bash
python -m datacol.annotate_pkl "pilot_sessions/$ID_R01"
python -m datacol.annotate_pkl "pilot_sessions/$ID_R02"
```

Fluxo para cada gesto:

1. Vá ao primeiro quadro.
2. Pressione `b`.
3. Vá ao último quadro.
4. Pressione:
   - `1`: `get_connectors`;
   - `2`: `get_screws`;
   - `3`: `get_wheels`;
   - `i`: `ignore`.

No bloco dos quatro tubos curtos de `four_tubes`:

1. Marque o bloco inteiro como `ignore`.
2. Volte ao primeiro quadro desse bloco.
3. Pressione `f` ou clique em `F FOUR_TUBES`.
4. Confirme no painel o número do quadro marcado.

Pressione `s` para salvar.

Cada sessão deve gerar:

```text
annotations.json
plan_events.json
```

Contagens esperadas por sessão:

```text
get_connectors: 8 intervalos
get_screws:    12 intervalos
get_wheels:     4 intervalos
ignore:         9 intervalos
begin_four_tubes: 1 evento
```

## 8. Gerar o dataset

```bash
python -m datacol.build_json \
  --sessions-root pilot_sessions \
  --output datasets/v1/pilot_dataset.json \
  --report datasets/v1/pilot_report_classes.md \
  --test-session "$ID_R02" \
  --context-dim 7 \
  --window-size 5
```

R01 ficará em `train`; R02 ficará em `test`.

## 9. Analisar a coerência

Abra:

```text
datasets/v1/pilot_report_classes.md
datasets/v1/pilot_dataset.json
```

O teste está coerente quando:

1. O relatório contém janelas das quatro classes nos dois splits.
2. Nenhuma classe possui zero janelas.
3. `no_action` pode ser maior, mas não deve dominar de forma extrema.
4. R01 aparece somente em `train`.
5. R02 aparece somente em `test`.
6. Nenhuma janela possui intenção `ignore`.
7. Toda `pose` possui 5 linhas de 45 valores.
8. Todo `context` possui 7 valores entre 0 e 1.
9. O contexto final de ambas as sessões é:

```text
[1, 0, 0, 0, 1, 1, 1]
```

Interpretação:

```text
stage=None, 8/8 conectores, 12/12 parafusos, 4/4 rodas
```

Se R01 terminar corretamente e R02 não, revise a posição de
`begin_four_tubes`. Se uma classe tiver poucas janelas, aumente a duração do
gesto ou do repouso. Se houver muitas poses zeradas, melhore enquadramento e
iluminação antes da coleta definitiva.

Por fim, execute:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

O resultado esperado atualmente é:

```text
52 passed
```

Antes de transformar o piloto em coleta definitiva, consulte
[`session_status_2026-06-13.md`](session_status_2026-06-13.md). A política
temporal, o pré-processamento numérico e o loader de treinamento ainda precisam
ser fechados.
