# Planejamento de aquisição de dataset e fine-tuning (OS-5 a OS-7)

Este documento cobre o pipeline completo: os comandos de CLI de OS-1 a OS-4
(captura, anotação, auditoria de contexto, consolidação), o plano de
experimento para o fine-tuning com injeção de contexto (geração dos três
datasets, organização de resultados, mudanças propostas em `DLinear.py`,
protocolo de treino V0/V1/V2) e a integração em tempo real (OS-7). Substitui
o antigo `docs/guia_rapido_cli.md` (removido) e complementa
`docs/mapeamento_os1-4_vs_original.md`, que documenta as incompatibilidades
entre o pipeline novo e o original.

As seções 2 a 5 (fine-tuning) assumem as duas incompatibilidades já
mapeadas — ausência de entrada de contexto no `DLinear.py` original e
ausência da tarefa de trajetória (`posOutputs`) em `build_json.py` — e
propõem como endereçá-las. **Essa parte ainda não foi revisada tecnicamente
linha a linha contra o código atual**; trate como plano de trabalho, não
como especificação validada. A seção 1 (comandos de CLI) descreve scripts já
implementados e testados.

---

## 1. Comandos de CLI — OS-1 a OS-4 (aquisição e consolidação do dataset)

Fluxo completo, na ordem em que se usa:

```
capture_session.py → annotate_pkl.py → context_replay.py (opcional) → build_json.py
```

Ative o ambiente antes de qualquer comando:

```bash
cd /home/marcos-kalile/hrc-data-collection
source .venv/bin/activate
```

### 1.1 `capture_session.py` — gravar uma sessão (webcam + MediaPipe)

Para quê: grava vídeo + esqueleto sincronizados por `frame_idx`. Gera, dentro
de `sessions/<session_id>/`, os quatro artefatos `meta.json`, `frames.jsonl`,
`skeleton.pkl` e `video.mp4`.

**Passo 1 — descobrir câmeras disponíveis.** Lista só os índices OpenCV que
realmente abrem e entregam um quadro:

```bash
python -m datacol.capture_session --list-cameras
# index=2 resolution=1280x720
```

Por padrão testa os índices 0–9; para ampliar, use
`--camera-scan-limit 20`.

**Passo 2 — ver o próximo ID de sessão sugerido** (não inicia a captura, só
consulta):

```bash
python -m datacol.capture_session --suggest-session-id
# S07_20260708
```

O formato é `SNN_YYYYMMDD`, com `NN` sempre maior que o maior número já
usado, mesmo que a sessão anterior tenha outra data — evita colisão de ID.

**Passo 3 — gravar.** Forma recomendada, com ID e câmera automáticos
(usa a primeira câmera legível encontrada por `discover_cameras`):

```bash
python -m datacol.capture_session \
  --participant P01 \
  --script-id R01 \
  --camera-model "Intelbras WCI 1080p" \
  --camera-distance-m 2.2 \
  --width 1280 --height 720 --fps 30
```

Para fixar explicitamente o ID e o índice de câmera (útil em máquinas com
mais de uma câmera ou para reproduzir uma sessão específica):

```bash
python -m datacol.capture_session S07_20260708 \
  --participant P01 \
  --script-id R01 \
  --camera-model "Intelbras WCI 1080p" \
  --camera-distance-m 2.2 \
  --camera-index 2 \
  --width 1280 --height 720 --fps 30
```

Opções úteis:

| Flag | Efeito |
|---|---|
| `--max-frames N` | encerra automaticamente após N quadros (útil para testes rápidos) |
| `--no-preview` | grava sem abrir janela de vídeo |
| `--autofocus <valor>` | registra a configuração de foco usada (ex. `locked_v4l2`) |
| `--camera-height chest` | registra a altura da câmera no metadado |
| `--zone-layout-version v1` | registra a versão do layout das zonas de pega |
| `--output-root <dir>` | muda o diretório onde as sessões são criadas (padrão `sessions`) |
| `--quaternion-world W X Y Z` | sobrescreve o quaternion registrado (não é aplicado à captura, só documentado) |

Durante a gravação, a janela mostra `REC <frame>`, tempo decorrido, FPS
observado e `POSE OK`/`POSE MISSING`. `q` ou `Esc` encerram normalmente;
`Ctrl+C` também encerra preservando os quadros já gravados. Um diretório de
sessão existente nunca é sobrescrito — use outro ID para gravar de novo.

**Passo 4 — validar a sessão.** A captura já valida automaticamente ao
final; para repetir manualmente (por exemplo, depois de mover os arquivos):

```bash
python -m datacol.capture_session --validate-session sessions/S07_20260708
```

Saída `0` (sessão válida) ou `1` (inválida, com a lista de erros impressa em
JSON).

### 1.2 `annotate_pkl.py` — rotular a sessão gravada

Para quê: abre o vídeo com o esqueleto sobreposto, permite marcar intervalos
de intenção e o evento `begin_four_tubes`. Gera `annotations.json` e
`plan_events.json` dentro da própria pasta da sessão.

```bash
python -m datacol.annotate_pkl sessions/S07_20260708
```

Fluxo recomendado dentro da interface:

1. Use `Espaço` para localizar aproximadamente o gesto, navegando com
   `a`/`d` (±1 quadro) e `j`/`l` (±10 quadros), ou clique direto na linha do
   tempo colorida.
2. No primeiro quadro do gesto, pressione `b` (ou clique em
   `MARCAR INICIO`).
3. Navegue até o último quadro do gesto.
4. Pressione a tecla da classe (`1`=`get_connectors`, `2`=`get_screws`,
   `3`=`get_wheels`, `i`=`ignore`, `0`=`no_action`/desmarcar) — o intervalo
   inteiro recebe o rótulo.
5. Repita para os demais gestos e para trechos `ignore`.
6. No primeiro quadro do bloco `ignore` que representa a entrega dos quatro
   tubos curtos, pressione `f` para marcar `begin_four_tubes`.
7. Pressione `s` para validar, salvar e encerrar.

Tabela de controles:

| Tecla | Ação |
|---|---|
| `Espaço` | reproduzir / pausar |
| `a` ou seta esquerda | voltar 1 quadro |
| `d` ou seta direita | avançar 1 quadro |
| `j` | voltar 10 quadros |
| `l` | avançar 10 quadros |
| `b` | marcar início do intervalo no quadro atual (pressionar de novo cancela a marca) |
| `0`,`1`,`2`,`3`,`i` | aplica a classe do início marcado até o quadro atual |
| `f` | marca ou remove `begin_four_tubes` no quadro atual |
| `u` | desfaz a última aplicação de classe feita nesta execução |
| `s` | valida, salva e encerra |
| `q` ou `Esc` | cancela sem salvar |

`no_action` cobre todos os quadros por padrão — não é preciso marcar
manualmente os trechos de repouso, só as ações e os trechos `ignore`. É
permitido marcar o fim do intervalo antes do início e navegar para trás: o
programa ordena os dois limites automaticamente. Se `annotations.json` já
existir e for válido, ele é recarregado ao reabrir a mesma sessão, permitindo
continuar ou revisar uma anotação incompleta.

Antes de salvar, o programa exige cobertura total (todo quadro pertence a
exatamente um rótulo, sem lacunas nem sobreposição) e que `begin_four_tubes`
esteja marcado no primeiro quadro de um bloco `ignore` — se alguma dessas
condições falhar, `s` recusa salvar e mantém a sessão aberta para correção.

### 1.3 `context_replay.py` — auditar o contexto antes de consolidar (opcional)

Para quê: reproduz vídeo + anotações + estado do `PlanGraph` lado a lado,
para conferir visualmente se as transições de estágio e os contadores de
contexto (7D/10D) estão coerentes antes de gastar tempo rodando
`build_json.py` sobre uma anotação com problema.

```bash
# abrir a interface interativa, começando do quadro 0, contexto 7D
python -m datacol.context_replay sessions/S07_20260708 --context-dim 7

# abrir pausado em um quadro específico
python -m datacol.context_replay sessions/S07_20260708 \
  --context-dim 10 --start-frame 1200 --paused

# só exportar um vídeo MP4 do replay, sem abrir janela (útil para revisão em lote)
python -m datacol.context_replay sessions/S07_20260708 \
  --export-only --output-video sessions/S07_20260708/context_replay.mp4
```

Use isso principalmente depois de anotar sessões com `begin_four_tubes`, para
confirmar visualmente que o estágio muda no momento certo antes de
consolidar.

### 1.4 `build_json.py` — consolidar sessões anotadas no dataset

Para quê: varre `--sessions-root`, junta todas as sessões anotadas
(precisam ter `meta.json`, `skeleton.pkl` e `annotations.json`) em um único
JSON (`split → intention → session → {start, end, windows}`), gera janelas
de pose `[window_size, 45]` com o contexto do `PlanGraph` já calculado,
separa treino/teste por sessão inteira (nunca por frame) e escreve um
relatório de classes em Markdown.

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

Detalhes de cada flag:

| Flag | Efeito |
|---|---|
| `--sessions-root` | diretório que contém as pastas de sessão (padrão `sessions`) |
| `--output` | caminho do JSON consolidado gerado |
| `--report` | caminho do relatório Markdown de contagem de classes |
| `--test-session <ID>` | reserva essa sessão inteira para teste; pode repetir a flag várias vezes; qualquer sessão anotada fora da lista vai para treino |
| `--context-dim {0,7,10}` | dimensão do vetor de contexto do `PlanGraph` embutido em cada janela |
| `--window-size` | número de quadros por janela (padrão 5, o contrato do modelo) |
| `--allow-empty-split` | permite treino ou teste vazio — só para ensaios técnicos com uma única sessão, nunca no dataset experimental final |

Regras de elegibilidade e validação:

- É preciso pelo menos 2 sessões anotadas (uma para treino, outra para
  teste); IDs de `--test-session` desconhecidos, repetidos, ou um split
  vazio sem `--allow-empty-split` fazem o comando falhar em vez de dividir
  silenciosamente de forma incorreta.
- `session_id` dentro de `meta.json` precisa bater com o nome da pasta.
- Intervalos rotulados `ignore` são excluídos das janelas; janelas que
  cruzam a fronteira entre dois rótulos diferentes também são descartadas.

Ao final, o terminal imprime os caminhos gerados:

```
dataset: datasets/v1/dataset.json
report: datasets/v1/report_classes.md
```

Abra o `report_classes.md` para conferir a contagem de janelas por classe e
split antes de considerar o dataset pronto para uso.

### 1.5 Referência rápida — para que serve cada arquivo-fonte

| Arquivo | Papel |
|---|---|
| `src/datacol/capture_session.py` | Logger de captura (webcam + MediaPipe Pose). Grava os 4 artefatos de uma sessão e valida integridade. |
| `src/datacol/joints15.py` | Mapeamento fixo dos 33 landmarks MediaPipe → 15 juntas do modelo (`joints15`). Usado pela captura. |
| `src/datacol/annotate_pkl.py` | Interface gráfica de anotação quadro a quadro. Produz `annotations.json` e `plan_events.json`. |
| `src/datacol/plan_sim.py` | Simulador offline do `PlanGraph` (estado da montagem): decide/confirma ações e calcula o vetor de contexto 7D/10D. Usado por `build_json.py` e `context_replay.py`. |
| `src/datacol/context_replay.py` | Auditoria visual do contexto: replay de vídeo + anotação + estado do `PlanGraph`. |
| `src/datacol/build_json.py` | Consolida sessões anotadas em um dataset único (JSON) e gera relatório de classes. |
| `src/datacol/__init__.py` | Contratos públicos: `INTENTION_LIST` e `ANNOTATION_LABELS`. |

### 1.6 Testes (por módulo)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_integrity.py     # OS-1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_annotate.py      # OS-2
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_plan_sim.py      # OS-3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_build_json.py    # OS-4
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q                             # tudo
```

O bloqueio de plugins evita que plugins pytest do ROS global contaminem este
repositório independente.

---

## 2. Datasets de contexto: `context_dim` = 0, 7 e 10

`plan_sim.py`/`build_json.py` já expõem os três `context_dim` sem qualquer
alteração de código:

- **`0`** — baseline sem contexto (vetor `[]`).
- **`7`** — one-hot de estágio (`none`/`bottom`/`four_tubes`/`top`) + 3
  contadores agregados (conectores/8, parafusos/12, rodas/4).
- **`10`** — mesmo one-hot + 6 contadores desagregados (tubos curtos/longos,
  parafusos por estágio, rodas).

### Organização de pastas

```
datasets/
  v1/
    dataset_dim0.json        # usado por V0 (avaliação) e V1 (fine-tune)
    dataset_dim7.json        # usado por V2_dim7
    dataset_dim10.json       # usado por V2_dim10
    report_classes_dim0.md
    report_classes_dim7.md
    report_classes_dim10.md
    manifest.json             # comandos exatos usados no build_json.py

runs/
  V0/
    eval_metrics.json          # avaliação do checkpoint original, sem treino
  V1/
    seed0/{checkpoint.pth, metrics.json, train_log.txt}
    seed1/...
    seed2/...
    config.json
  V2_dim7/
    seed0/... seed1/... seed2/...
    config.json
  V2_dim10/
    seed0/... seed1/... seed2/...
    config.json

results/
  ablation_table.md            # agregado de runs/*/seed*/metrics.json
```

### Comando de geração (mesmo split nos três)

```bash
for dim in 0 7 10; do
  python -m datacol.build_json \
    --sessions-root sessions \
    --output datasets/v1/dataset_dim${dim}.json \
    --report datasets/v1/report_classes_dim${dim}.md \
    --test-session S09_20260620 --test-session S10_20260620 \
    --context-dim ${dim} \
    --window-size 5
done
```

**Ponto crítico:** rodar as três variações com **exatamente o mesmo
`--sessions-root` e a mesma lista de `--test-session`**, para garantir que o
split treino/teste seja idêntico entre os três JSONs — a única diferença
entre eles deve ser o vetor de contexto. Guardar os três comandos exatos em
`manifest.json`.

**Checagem de sanidade antes de treinar:** comparar
`report_classes_dim7.md`/`dim10.md` contra `dim0.md` — a contagem de janelas
por classe/split deve bater exatamente nos três relatórios. Se não bater,
algo no janelamento divergiu entre as rodadas (não deveria acontecer, já que
só `--context-dim` muda).

| Dataset | Consumido por |
|---|---|
| `dataset_dim0.json` | V0 (avaliação) e V1 (fine-tune, `context_dim=0`) |
| `dataset_dim7.json` | V2_dim7 (fine-tune a partir de V1) |
| `dataset_dim10.json` | V2_dim10 (fine-tune a partir de V1) |

---

## 3. Mudança proposta em `DLinear.py`: camada de injeção de contexto

Proposta: injetar o contexto como termo residual no vetor latente que
alimenta `Intention_Predictor` em `Model_FinalIntention`, com projeção
linear **zero-inicializada** para não perturbar o comportamento do checkpoint
original no início do fine-tuning:

```python
class Model_FinalIntention(nn.Module):
    def __init__(self, args, context_dim: int = 0):
        super().__init__()
        # ... arquitetura original sem alteração ...
        self.context_dim = context_dim
        if context_dim > 0:
            latent_dim = self.seq_len * self.class_num
            self.context_proj = nn.Linear(context_dim, latent_dim)
            nn.init.zeros_(self.context_proj.weight)
            nn.init.zeros_(self.context_proj.bias)

    def forward(self, x, context=None):
        # ... seasonal_init, trend_init, traj_output iguais ao original ...
        latent = intention_vector.reshape(x.shape[0], -1)
        if self.context_dim > 0 and context is not None:
            latent = latent + self.context_proj(context)
        intention_output = self.Intention_Predictor(latent)
        return traj_output, intention_output
```

### Reaproveitamento de pesos entre variantes

```python
# V0 -> V1: mesma arquitetura (context_dim=0 nos dois)
model_v1 = Model_FinalIntention(args, context_dim=0)
model_v1.load_state_dict(torch.load("checkpoints/original.pth"), strict=False)

# V1 -> V2: arquitetura ganha context_proj (zero-init no __init__)
model_v2 = Model_FinalIntention(args, context_dim=7)  # ou 10
model_v2.load_state_dict(torch.load("runs/V1/seed0/checkpoint.pth"), strict=False)
# strict=False ignora a ausência de "context_proj.*" no state_dict de V1;
# como context_proj já nasce zerado, o forward de V2 no passo 0 é
# numericamente idêntico ao de V1.
```

**Teste de aceite recomendado antes de treinar de fato:** rodar o mesmo batch
em `model_v1` e em `model_v2` recém-carregado (qualquer `context`) e conferir
que `intention_output` bate exatamente — confirma que zero-init +
`strict=False` funcionam antes de gastar horas de treino sobre um bug
silencioso.

---

## 4. Protocolo de fine-tuning (V0 / V1 / V2)

Esqueleto de função de treino (interface, não implementação final):

```python
def train_variant(variant: str, context_dim: int, dataset_path: str,
                   init_checkpoint: str | None, seed: int, out_dir: Path):
    torch.manual_seed(seed)
    model = Model_FinalIntention(args, context_dim=context_dim)
    if init_checkpoint:
        model.load_state_dict(torch.load(init_checkpoint), strict=False)

    dataset = load_dataset(dataset_path, context_dim=context_dim)
    train_loader, test_loader = split_loaders(dataset)

    # Fase 1: backbone congelado, só context_proj + Intention_Predictor treinam
    freeze_backbone(model)
    optimizer = build_optimizer(model, phase=1, lr_head=1e-4)
    run_epochs(model, train_loader, optimizer, n_epochs=4)

    # Fase 2: descongela tudo, LR discriminativo
    unfreeze_backbone(model)
    optimizer = build_optimizer(model, phase=2, lr_backbone=1e-5, lr_head=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=...)
    run_epochs(model, train_loader, optimizer, scheduler, n_epochs=6)

    metrics = evaluate(model, test_loader)  # top-1, por classe, matriz confusão, ECE
    save_run(out_dir, model, metrics, config={
        "variant": variant, "context_dim": context_dim, "seed": seed,
        "dataset": dataset_path, "init_checkpoint": init_checkpoint,
    })
```

### Ordem de execução

```
V0:        avalia checkpoint original direto em dataset_dim0.json (test split) -> runs/V0/eval_metrics.json
V1:        for seed in [0,1,2]: train_variant("V1", context_dim=0, dataset_dim0, init=original, seed)
V2_dim7:   for seed in [0,1,2]: train_variant("V2_dim7",  7,  dataset_dim7,  init=runs/V1/seed{seed}, seed)
V2_dim10:  for seed in [0,1,2]: train_variant("V2_dim10", 10, dataset_dim10, init=runs/V1/seed{seed}, seed)
```

**Protocolo de seed pareado:** cada seed de V2 inicializa a partir do
**mesmo índice de seed** de V1 (`V2_dim7/seed1` vem de `V1/seed1`, não de um
único V1 "campeão"). Isso propaga a variância de inicialização de forma
pareada pelas três variantes, deixando o delta V2−V1 estatisticamente mais
honesto — é o que sustenta comparar médias ± desvio-padrão no final.

Ao final, um script de agregação lendo `runs/*/seed*/metrics.json` monta a
tabela V0/V1/V2_dim7/V2_dim10 para o artigo (`results/ablation_table.md`).

---

## 5. OS-7 — quando o vetor de contexto deve ser atualizado em tempo real

**Não atualizar a cada predição bruta do modelo.** Se cada frame por si só
disparasse a atualização do `PlanGraph`, uma predição ruidosa isolada (alta
entropia, gesto ambíguo) corromperia o estado de forma permanente —
contadores e estágio são progresso cumulativo, não devem regredir/avançar
por engano.

O ponto correto é o mesmo bloco de **confirmação por janela** que já existe
em `run_webcam.py` (frames consecutivos concordando, onde hoje está
`send_intention(intention)`). É esse evento discreto — não cada frame — que
deve mutar o `PlanGraph` e recalcular o contexto:

```python
# no início de run_live(), antes do loop:
plan = PlanGraph(policy="proxy_graph")
context_dim = args.context_dim  # 0, 7 ou 10, deve bater com o checkpoint carregado
cached_context = plan.to_context_vector(dim=context_dim) if context_dim else None

# dentro do loop, na predição:
context_tensor = torch.tensor(cached_context).float().unsqueeze(0) if cached_context is not None else None
_, pred_intention = predictor.predict(inputs, restrict=restrict, context=context_tensor)

# no bloco de confirmação, onde send_intention(intention) já é chamado:
if intention and intention != 'no_action' and confirmado_por_send_window:
    send_intention(intention)
    plan.step(intention)
    cached_context = plan.to_context_vector(dim=context_dim)  # recalcula só aqui
```

O vetor fica em cache e só é recalculado no evento de confirmação — todos os
frames entre uma confirmação e a próxima usam o mesmo vetor. Isso preserva a
mesma semântica de amostragem do `build_json.py` (contexto vigente *antes*
da ação, atualizado só depois que ela é dada como concluída).

### Pontos de atenção específicos deste pipeline (webcam, sem robô, sem voz)

- **Sem ROS de fato:** `run_webcam.py` já roda em processo único com as
  linhas de ROS comentadas — OS-7 aqui não precisa de publisher/subscriber,
  só do objeto `PlanGraph` local no escopo de `run_live()`. Não é necessário
  tocar em `controller/receiver.py`.
- **`four_tubes` sem gatilho automático:** no sistema original, quem
  dispara isso é o comando de voz `short` (via `speech_recognize.py`); no
  demo de webcam não há reconhecedor de voz rodando. É preciso um gatilho
  manual equivalente ao `f` do `annotate_pkl.py` — por exemplo, uma tecla no
  loop do `run_webcam.py` que chama `plan.begin_four_tubes_stage()` e
  recalcula `cached_context` na hora. Documentar isso como limitação
  conhecida da demonstração online.
- **Cold start:** antes da primeira confirmação, `cached_context` deve ser o
  vetor do estado inicial (`stage=none`, contadores zerados) — não `None`
  nem zeros arbitrários — para bater com o que o modelo viu nas primeiras
  janelas de cada sessão durante o treino.
