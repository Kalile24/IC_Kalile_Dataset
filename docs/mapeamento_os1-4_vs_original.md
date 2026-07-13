# Mapeamento OS-1 a OS-4 vs. repositório original (Yu et al.)

Este documento resume a comparação entre os módulos implementados em
`hrc-data-collection` (OS-1 a OS-4) e seus equivalentes/precursores no
repositório original `IC_Kalile_Intention_Prediction_HRC`. Não substitui a
leitura dos módulos individuais (`docs/capture_session.md`,
`docs/annotate_pkl.md`, `docs/plan_sim.md`, `docs/build_json.md`); é um mapa de
alto nível para orientar as decisões de OS-5/OS-6/OS-7.

## Tabela de correspondência

| Arquivo novo | Arquivo/função original correspondente | Preservado | Mudado e por quê |
|---|---|---|---|
| `capture_session.py` | `run_webcam.py` (captura + `mediapipe_fallback.py`) | Mapeamento MediaPipe→15 joints (`range(11,25)+(0,)`); API `mp.solutions.pose.Pose`; uso de OpenCV para webcam (já presente em `run_webcam.py`, ausente em `run.py`/OAK-D); quaternion `_CAM_TO_WORLD_Q` registrado com o mesmo valor numérico | Nenhum ROS (nunca existiu, nem comentado); persistência em 4 artefatos sincronizados por `frame_idx` (`meta.json`, `frames.jsonl`, `skeleton.pkl`, `video.mp4`); grava `landmarks_raw` (33,4) além de `joints15` (15,3) para permitir reprocessamento futuro; `smooth_landmarks=False` (original usa `True`) para não perder dado bruto; quaternion registrado mas **não aplicado** na captura — rotação vira responsabilidade do pré-processamento offline; validação de integridade pós-captura (`validate_session`), inexistente no original |
| `annotate_pkl.py` | Nenhum equivalente direto — no original, os rótulos eram a *saída* do controlador em tempo real (`decide_send_action`/`Receiver`), não um dado anotado offline | Classes de intenção idênticas a `INTENTION_LIST` do `Dataset.py` original (`no_action`, `get_connectors`, `get_screws`, `get_wheels`); estilo visual do overlay de esqueleto segue `run_webcam.py` | Arquitetura inteira é nova: rotulagem humana offline substitui decisão em tempo real, porque não há robô real reagindo a percepção ao vivo; introduz `ignore` como quinto rótulo (não é classe de intenção, é marca de exclusão do dataset); introduz `plan_events.json`/`begin_four_tubes` como evento externo separado das classes de intenção |
| `plan_sim.py` (`PlanGraph`, política `"receiver"`) | `controller/receiver.py` (`PlanGraph` + `Receiver.decide_send_action`) | Lógica de decisão replicada quase linha a linha (`_decide_action` ≈ `decide_send_action`): condições de `bottom`/`top`, reversão de tubo, contagem de parafusos por estágio, liberação de rodas, comandos `short`/`long`/`spin`/`get up` | Separa explicitamente decisão (`apply_intention`/`apply_command`) de confirmação de execução (`apply_action`) — no original essas duas fases estão fundidas em `decide_send_action` (que já muta parafusos) + `execute_action` (que muta tubos/estágios só ao fim do movimento físico); não há ROS, não há waypoints, não há atuação física — puro replay de estado |
| `plan_sim.py` (`PlanGraph`, política `"proxy_graph"`) | Mesma base do `receiver.py`, mas corrigindo dois comportamentos específicos | Estrutura geral do grafo (dois caminhos `bottom→top→four_tubes` e `bottom→four_tubes→top`) | Corrige contagem de parafusos por estágio: uniformiza para 4 parafusos por estágio, sempre nomeados corretamente (`spin_{stage}`), em vez do comportamento assimétrico do legado (`receiver.py:590-601`, que permite até 8 parafusos em `top` e retorna `"spin_bottom"` mesmo em estágio `top`, dependendo da ordem de conclusão); corrige liberação de rodas para exigir os 12 parafusos completos (`all(count==4 ...)`) em vez do rebaixamento prematuro do legado (`spin_count < 8` em `receiver.py:569`/`plan_sim.py:305-307`) |
| `plan_sim.py` (`begin_four_tubes_stage`/evento `begin_four_tubes` anotado) | `speech/speech_recognize.py` (comando de voz `"short"`) + trecho especial de `receiver.py:110-119` (`receive_data`, fora de `decide_send_action`) | Mesmas três condições de ativação do estágio (`stage is None`, `bottom` completo, `four_tubes` não completo) | **Substituição de modalidade, não reimplementação**: no original, o estágio `four_tubes` é ativado por um comando de voz reconhecido em tempo real (PyAudio → VAD → DeepSpeech → heurísticas difusas de correção → `"short"` publicado no tópico ROS `chatter`). No novo pipeline não há STT em lugar nenhum; o "sinal" que ativa `four_tubes` passa a ser a marcação manual do humano anotador (tecla `f` em `annotate_pkl.py`, gravada em `plan_events.json`), consumida por `build_json.py` via `_apply_plan_event`/`begin_four_tubes_stage`. O canal de entrada inteiro (voz) foi trocado por rotulagem humana offline |
| `sender.py` (original) | — | — | Sem equivalente em `plan_sim.py`: `sender.py` existe só para publicar strings manualmente no tópico ROS `chatter` como ferramenta de teste do `Receiver`. Sem ROS, a chamada direta de função (`apply_intention`/`apply_command`) já é o mecanismo de entrega — não há barramento a simular |
| `build_json.py` | `traj_intention/Dataset.py` (`MyDataset.process_json`) | Hierarquia `split → intention → session → {start, end}` preservada intencionalmente para compatibilidade com o loader legado; `INTENTION_LIST` idêntico | Converte índices zero-based/inclusivos (OS-2) para 1-based/exclusivos (formato que `Dataset.py` espera, linha `end_index = end-1`); adiciona campo `windows` com pose já achatada `[5,45]` e contexto, não usado pelo loader legado mas destinado a um loader novo do OS-5; contexto é amostrado *antes* da conclusão da ação anotada (previne vazamento de rótulo); grava `plan_policy: "proxy_graph"` em `_meta`, divergindo deliberadamente do `receiver.py` legado |

## Incompatibilidade crítica para OS-5

`Dataset.py` original (`traj_intention/Dataset.py:81-89`) espera que `.pkl`
contenha uma sequência de **objetos** com atributo `.landmarks` (e
opcionalmente `.landmarks_world`, `.xyz`, `.score` — o contrato do
`Body`/`FakeBody` visto em `mediapipe_fallback.py`). O novo `skeleton.pkl`
(OS-1) é um `np.ndarray` puro `(N, 15, 3)`. O branch `input_type == "pkl"` do
loader legado falharia (`AttributeError`) se apontado diretamente para o novo
`skeleton.pkl`. O campo `windows` em `build_json.py` existe para contornar
essa incompatibilidade, mas exige um loader novo (ainda não escrito) — não um
reaproveitamento do `Dataset.py` como está.

Além disso, `Dataset.py` aplica normalização min-max por-janela e
`camera_to_world` no momento de montar cada exemplo de treino; nenhuma das
janelas gravadas por `build_json.py` recebe esse pré-processamento — ele fica
pendente para o loader/pipeline do OS-5.

## Incompatibilidade central para OS-5/6/7: `DLinear.py`/`train.py` não recebem contexto

`Model_FinalIntention` e `Model_FinalTraj` (`traj_intention/DLinear.py:39-95,
96-156`) têm `forward(self, x)` recebendo **apenas a pose** `[Batch, seq_len,
channels]`. A predição de intenção é derivada só de pose:

- `Model_FinalIntention`: `Intention_Predictor = nn.Linear((seq_len+pred_len)*channels, class_num)`
  (linha 69), aplicado à concatenação de pose de entrada e trajetória prevista.
- `Model_FinalTraj`: `Intention_Vector = nn.Linear(channels, class_num)`
  (linha 126), aplicado diretamente a `x` (a pose).

`train.py:79` confirma: `pred_traj, pred_intention = net(inputs)`, onde
`inputs` vem de `Dataset.py` como apenas a janela de pose — nunca um vetor de
contexto do `PlanGraph`.

Isso significa que **o vetor de contexto 7D/10D produzido em `plan_sim.py` e
gravado em cada janela de `build_json.py` não tem nenhum ponto de entrada no
modelo original**. "Fine-tuning com injeção de contexto" (OS-5/6/7) exige
modificar a arquitetura do DLinear (ou escrever uma cabeça nova) para
concatenar esse vetor em algum ponto do forward — não é só um ajuste de
loader de dados, é uma mudança na definição do modelo.

Isso tem uma implicação direta para o desenho experimental: qualquer
arquitetura com contexto deixa de ser a mesma rede avaliada por Yu et al., de
modo que uma diferença de acurácia observada passa a misturar três efeitos
que não podem ser separados sem um baseline controlado: (1) o efeito do
contexto em si, (2) a mudança arquitetural necessária para aceitá-lo, e (3) as
divergências de dataset já mapeadas acima (webcam vs. OAK-D, `proxy_graph` vs.
`receiver`). Recomenda-se que o OS-5 trate isso como comparação interna
controlada (DLinear sem contexto vs. DLinear+contexto, ambos treinados nos
mesmos dados novos) em vez de comparação direta com os números publicados no
paper original.

## Incompatibilidade adicional: estrutura de treino multi-tarefa (trajetória + intenção)

`Dataset.py` (`traj_intention/Dataset.py:105-127`) monta cada exemplo de
treino como uma tupla `(posInputs, posOutputs, intention_label)`:

- `posInputs = npy_file[j : j+seq_len]` — janela de entrada (`seq_len=5`
  frames), equivalente ao campo `pose` de uma `window` em `build_json.py`.
- `posOutputs = npy_file[j+seq_len : j+seq_len+pred_len]` — os `pred_len=5`
  frames **seguintes**, usados como alvo de regressão da trajetória futura.

`train.py:79-82` usa as duas saídas do modelo simultaneamente:

```python
pred_traj, pred_intention = net(inputs)
traj_loss = traj_criterion(pred_traj, target_traj)       # MSELoss
intention_loss = class_criterion(pred_intention, labels)  # CrossEntropyLoss
loss = traj_loss + intention_loss
```

O treinamento original é portanto **multi-tarefa**: o `DLinear`
(`Model_FinalIntention`/`Model_FinalTraj`) aprende a prever a continuação da
trajetória e a classificar a intenção ao mesmo tempo, com uma loss somada.

`build_json.py` (`build_windows`, linhas 68-134) só materializa a janela de
entrada e o rótulo de classe — não existe campo equivalente a `posOutputs`
nem preocupação em reservar os `pred_len` frames seguintes como alvo de
regressão. O dataset novo, como está, **não sustenta a tarefa de trajetória**
do `train.py` original, só a classificação de intenção.

Isso se soma à amostragem negativa: `Dataset.py` (linhas 129-141) sorteia
exemplos de `no_action` fora dos intervalos anotados por classe, em
quantidade calculada para balancear classes
(`pos_data_num = len(task_data) // len(intention_list) // 2`).
`build_json.py` não replica esse sorteio/balanceamento — usa apenas os
trechos já rotulados como `no_action` na anotação (Módulo 2), sem controle
de proporção por classe.

Isso deixa uma decisão de escopo em aberto para o OS-5: (a) reimplementar a
lógica de pares `posInputs`/`posOutputs` e o balanceamento de negativos num
builder novo, preservando a arquitetura multi-tarefa original; ou
(b) simplificar deliberadamente para treino de classificação pura,
descartando a cabeça de trajetória — documentando essa simplificação como
mudança de escopo em relação ao paper original, e não como equivalência.

## Limitação de comparabilidade com o original

O dataset gerado usa a política `proxy_graph` do `PlanGraph`, que corrige duas
divergências do `receiver.py` legado (contagem de parafusos por estágio e
condição de liberação de rodas). Isso significa que o vetor de contexto (7D
ou 10D) usado para treinar/fine-tunar o modelo **não é diretamente comparável**
ao comportamento do controlador físico do experimento original de Yu et al.
Essa é uma escolha deliberada de corrigir o grafo em vez de replicar o legado
fielmente, e deve ser documentada explicitamente em qualquer comparação de
resultados com o paper original.
