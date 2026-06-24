# Guia do `annotate_pkl.py`

O `annotate_pkl.py` implementa a OS-2: reproduz uma sessao quadro a quadro,
desenha `skeleton.pkl` sobre `video.mp4`, permite marcar intervalos por teclado
e grava `annotations.json`.

## Pre-requisitos

A sessao deve ter sido concluida pela OS-1 e conter:

```text
sessions/S01_20260620/
├── meta.json
├── frames.jsonl
├── skeleton.pkl
└── video.mp4
```

Antes de anotar, e recomendavel validar a sessao:

```bash
python -m datacol.capture_session \
  --validate-session sessions/S01_20260620
```

## Iniciar

Com a `.venv` ativa:

```bash
python -m datacol.annotate_pkl sessions/S01_20260620
```

O anotador verifica que vídeo e esqueleto possuem a mesma quantidade de
quadros. O vídeo e lido sob demanda, mantendo apenas um quadro em cache para
nao carregar a sessao inteira na memoria.

Se `annotations.json` ja existir e for valido, ele e recarregado para
continuar ou revisar a anotacao. Caso contrario, todos os quadros comecam como
`no_action`.

## Interface grafica

A janela e dividida em duas partes:

- vídeo com o esqueleto sobreposto, à esquerda;
- painel de anotação clicável, à direita.

O painel mostra o quadro atual, a classe vigente, o início marcado e uma linha
do tempo colorida por classe. Cada trecho usa a mesma cor de seu botão; a
linha branca indica o quadro atual e a marca branca adicional indica o início
do intervalo selecionado. Todos os controles principais podem ser usados com
o mouse:

- clique na linha do tempo para saltar para outro quadro;
- use `<<10`, `<1`, `PLAY`, `1>` e `10>>` para navegar;
- clique em `MARCAR INICIO` no primeiro quadro do intervalo;
- use `F FOUR_TUBES` no primeiro quadro do bloco `ignore` que entrega os
  quatro tubos curtos;
- clique em `CANCELAR MARCA` para abandonar uma seleção ainda não aplicada;
- clique no botão colorido da classe no último quadro;
- use `DESFAZER` para reverter a última aplicação;
- clique em `SALVAR` ou `CANCELAR` para encerrar.

Os atalhos de teclado continuam disponíveis para anotação rápida.

## Classes

| Tecla | Rotulo | ID do modelo |
|---|---|---:|
| `0` | `no_action` | 0 |
| `1` | `get_connectors` | 1 |
| `2` | `get_screws` | 2 |
| `3` | `get_wheels` | 3 |
| `i` | `ignore` | nenhum |

`ignore` e salvo explicitamente, mas sera excluido do dataset pela OS-4.
O evento `begin_four_tubes` não é uma classe: ele é salvo separadamente em
`plan_events.json`.

## Controles

| Tecla | Acao |
|---|---|
| `Espaco` | reproduzir ou pausar |
| `a` ou seta esquerda | voltar 1 quadro |
| `d` ou seta direita | avancar 1 quadro |
| `j` | voltar 10 quadros |
| `l` | avancar 10 quadros |
| `b` | marcar inicio do intervalo no quadro atual |
| `f` | marcar ou remover `begin_four_tubes` no quadro atual |
| `0`, `1`, `2`, `3`, `i` | aplicar classe do inicio marcado ate o quadro atual |
| `u` | desfazer a ultima aplicacao feita nesta execucao |
| `s` | validar, salvar e encerrar |
| `q` ou `Esc` | cancelar sem salvar alteracoes |

Os limites sao zero-based e inclusivos em `annotations.json`.

## Fluxo recomendado

1. Use `Espaco` para localizar aproximadamente o gesto.
2. Pause e navegue com `a`, `d`, `j` e `l`.
3. No primeiro quadro do gesto, pressione `b`.
4. Navegue ate o ultimo quadro do gesto.
5. Pressione a tecla da classe.
6. Repita para os demais gestos e para trechos `ignore`.
7. No primeiro quadro do bloco `ignore` dos quatro tubos curtos, pressione
   `f`.
8. Pressione `s` para salvar.

Para classificar apenas o quadro atual, pressione a tecla da classe sem usar
`b`. Tambem e permitido marcar o fim primeiro e navegar para tras: o anotador
ordena os dois limites antes de aplicar o rotulo.

Como `no_action` cobre tudo inicialmente, nao e necessario marcar manualmente
os intervalos de repouso. Ao aplicar outra classe, ela substitui `no_action`
naquele intervalo.

## Marcar e desmarcar

Para marcar uma ação:

1. vá ao primeiro quadro;
2. clique em `MARCAR INICIO` ou pressione `b`;
3. vá ao último quadro;
4. clique na classe desejada ou use sua tecla.

Para desmarcar uma ação e devolver o trecho a `no_action`:

1. marque novamente o mesmo intervalo;
2. clique em `DESMARCAR / NO ACTION` ou pressione `0`.

Para cancelar apenas a seleção do início, sem alterar rótulos, clique em
`CANCELAR MARCA` ou pressione `b` novamente. Para desfazer a última aplicação
de classe desta execução, use `DESFAZER` ou `u`.

## Overlay

O esqueleto utiliza as 15 juntas na ordem `joints15 v1` e segue o estilo do
`run_webcam.py`:

- torso em azul;
- braços em vermelho;
- juntas verdes com contorno escuro.

A visualização é deliberadamente compacta: somente ombros, cotovelos, pulsos
e quadris são desenhados. Juntas de mãos e nariz continuam armazenadas nos
dados, mas não poluem o replay.

Quadros sem pose usam o sentinela zero da OS-1 e nao desenham um esqueleto
falso no canto da imagem.

## Arquivo gerado

O resultado segue `schemas/annotations.schema.json`:

```json
{
  "no_action": {
    "start": [0, 31],
    "end": [9, 40]
  },
  "get_connectors": {
    "start": [10],
    "end": [20]
  },
  "get_screws": {
    "start": [],
    "end": []
  },
  "get_wheels": {
    "start": [],
    "end": []
  },
  "ignore": {
    "start": [21],
    "end": [30]
  }
}
```

Quando `four_tubes` participa da sessão, o anotador também grava:

```json
{
  "events": [
    {
      "frame_idx": 1200,
      "event": "begin_four_tubes"
    }
  ]
}
```

Antes de salvar, o programa exige:

- exatamente as cinco chaves de rotulo;
- listas `start` e `end` com o mesmo tamanho;
- intervalos ordenados e dentro da sessao;
- nenhuma sobreposicao;
- cobertura de todo quadro por exatamente um rotulo;
- `begin_four_tubes` único e no primeiro quadro de um intervalo `ignore`.

## Testes

Execute os testes da OS-2:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_annotate.py
```

Eles verificam cobertura total, rejeicao de lacunas e sobreposicoes,
persistencia de `ignore`, round-trip JSON, navegacao reversa e overlay.

Para auditar o contexto derivado após a anotação, use
[`context_replay.md`](context_replay.md). As pendências antes da coleta
definitiva estão consolidadas em
[`session_status_2026-06-13.md`](session_status_2026-06-13.md).
