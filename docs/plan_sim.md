# Guia do `plan_sim.py`

O `plan_sim.py` implementa a OS-3: simula offline as transições do `PlanGraph`
de `controller/receiver.py`, sem importar ROS, executar waypoints ou controlar
o robô.

Depois de `bottom`, o grafo admite dois caminhos principais:

```text
bottom -> top -> four_tubes
bottom -> four_tubes -> top
```

O comando de voz `short` ativa o ramo `four_tubes`; as intenções visuais
`get_connectors` avançam `bottom` e `top`. Portanto, `four_tubes` não pode ser
deduzido apenas das quatro classes de intenção.

O código legado contém sobrescritas de `stage_history` marcadas como
`for debugging`. Elas evidenciam as duas ordens pretendidas, mas não formam
uma implementação confiável de todas as transições de uma montagem completa.

## Políticas

O simulador oferece:

```python
PlanGraph(policy="receiver")
PlanGraph(policy="proxy_graph")
```

`receiver` é o padrão e preserva as condições legadas e a equivalência com
`--stageI_done`.

`proxy_graph` mantém os dois caminhos, limita cada estágio a quatro parafusos
e libera rodas depois dos 12 parafusos. A OS-4 usa essa política porque o
legado converte uma intenção de parafuso em roda cedo demais em um caminho e
acumula oito parafusos em `top` no outro.

## Estado

`PlanGraph` mantem os mesmos componentes relevantes do controlador:

- contagem de tubos curtos e longos;
- contagem de parafusos em `bottom`, `four_tubes` e `top`;
- contagem de rodas;
- estágio ativo e histórico de estágios concluídos;
- ações concluídas por estágio;
- históricos de ações e intenções aceitas.

Um estágio de conectores termina depois de quatro ações de tubo. Nesse momento,
ele entra em `stage_history` e o estágio ativo volta a `None`.

## Decisão e conclusão

O controlador real decide uma ação antes de saber se sua execução terminou.
O simulador preserva essa separação:

```python
plan = PlanGraph()
action = plan.apply_intention("get_connectors")

if action is not None:
    action_name, previous_index = action
    plan.apply_action(action_name)
```

`apply_intention()` reproduz `decide_send_action`. `apply_action()` reproduz
somente a atualização feita depois de uma execução bem-sucedida. Portanto,
deixe de chamar `apply_action()` para simular uma ação decidida que falhou.

Para o replay de anotações, onde a ação observada e tratada como concluída,
use o atalho:

```python
action = plan.step("get_connectors")
```

`no_action` não altera o estado e retorna `None`.

## Momento de amostragem

Na OS-4, o intervalo anotado representa a intenção humana que antecede a ação
do controlador. Todas as janelas desse intervalo recebem o estado anterior à
ação. A transição é confirmada apenas depois do último quadro:

```python
context_during_interval = plan.to_context_vector(dim=7)
# ... emitir as janelas contidas no intervalo ...
plan.step(annotated_intention)
```

Assim, nenhuma janela usa como entrada um efeito provocado pelo próprio rótulo
que ela deve aprender a predizer. As janelas dos intervalos seguintes já
recebem o estado atualizado.

## Vetor 7D

`to_context_vector(dim=7)` retorna:

```text
[
  stage_none,
  stage_bottom,
  stage_four_tubes,
  stage_top,
  conectores_coletados / 8,
  parafusos_totais / 12,
  rodas / 4
]
```

Os quatro primeiros valores formam um one-hot do atributo `stage`.

## Vetor 10D

`to_context_vector(dim=10)` retorna:

```text
[
  stage_none,
  stage_bottom,
  stage_four_tubes,
  stage_top,
  tubos_curtos / 8,
  tubos_longos / 4,
  parafusos_bottom / 4,
  parafusos_four_tubes / 4,
  parafusos_top / 4,
  rodas / 4
]
```

Os valores normalizados são limitados ao intervalo `[0, 1]`. Isso preserva o
contrato do modelo mesmo se uma sequência fora do protocolo fizer o contador
legado ultrapassar o máximo experimental.

## Preset do estágio I

O construtor aceita o mesmo preset do receptor:

```python
plan = PlanGraph(stageI_done=True)
```

O estado inicial resultante possui:

- dois tubos curtos e dois longos;
- quatro parafusos em `bottom`;
- registro `bottom` com a sequência curto, longo, curto, longo;
- `stage_history == ["bottom"]`;
- nenhum estágio ativo.

Os testes também constroem esse estado pela sequência canônica de quatro
`get_connectors` e quatro `get_screws`, verificando equivalência de contadores,
registros e histórico de estágios.

## Estágio `four_tubes`

No receptor original, a ativação de `four_tubes` acontece fora de
`decide_send_action`, no tratamento do comando de voz `short`. Ela pode
ocorrer depois de `bottom`, antes ou depois da conclusão de `top`:

```python
plan.begin_four_tubes_stage()

for _ in range(4):
    action = plan.apply_command("short")
    if action is not None:
        plan.apply_action(action[0])
```

Também são aceitos os comandos manuais do decisor legado: `short`, `long`,
`spin` e `get up`. Eles não entram no histórico de intenções do modelo.

## Snapshot

`snapshot()` retorna uma cópia independente e serializável de todo o estado.
Ele e útil para depuração, testes de determinismo e relatórios da OS-4:

```python
state = plan.snapshot()
```

## Testes

Execute os testes da OS-3:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_plan_sim.py
```

Eles cobrem vetores 7D/10D, alternância de conectores, separação entre decisão
e conclusão, `four_tubes`, determinismo e equivalência com `--stageI_done`.

O simulador está validado para a OS-3 e é consumido diretamente por
`build_json.py` (política `proxy_graph`) na consolidação do dataset real.
