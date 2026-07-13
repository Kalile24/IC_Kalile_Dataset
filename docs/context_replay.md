# Auditoria gráfica do contexto

O `context_replay.py` permite conferir visualmente se a OS-3 reconstruiu o
estado correto da montagem em cada quadro. A ferramenta é somente de leitura:
ela não altera anotações nem o dataset.

## Executar

```bash
python -m datacol.context_replay \
  pilot_sessions/S01_20260613 \
  --context-dim 7
```

O vídeo começa reproduzindo automaticamente. Pressione `Espaço` para pausar.
Para abrir já pausado, acrescente `--paused`.

Para abrir diretamente em um quadro:

```bash
python -m datacol.context_replay \
  pilot_sessions/S01_20260613 \
  --context-dim 7 \
  --start-frame 1753
```

## O que aparece

- vídeo e esqueleto sincronizados;
- ação anotada no quadro atual;
- estágio ativo do `PlanGraph`;
- sete componentes do vetor com barras e valores numéricos;
- esquema da proxy com os três estágios, parafusos e rodas;
- linha do tempo colorida pelas classes.

O esquema usa a forma de um carrinho: trilho inferior (`bottom`), quatro
colunas (`four_tubes`), trilho superior (`top`), parafusos nas uniões e rodas.
As peças ganham cor conforme entram no estado do `PlanGraph`.

O desenho e o vetor são derivados pela mesma função usada no `build_json.py`.
O estado mostrado é sempre o vigente antes da conclusão da ação anotada. A
mudança causada por um gesto aparece no intervalo seguinte.

## Controles

| Controle | Ação |
|---|---|
| `Espaço` | reproduzir ou pausar |
| `a` ou seta esquerda | voltar 1 quadro |
| `d` ou seta direita | avançar 1 quadro |
| `j` | voltar 10 quadros |
| `l` | avançar 10 quadros |
| clique na linha do tempo | saltar para o ponto escolhido |
| botões `-10`, `-1`, `PLAY`, `+1`, `+10` | navegar com o mouse |
| botão `SAVE MP4` ou tecla `e` | salvar o replay completo |
| `q` ou `Esc` | fechar |

Por padrão, `SAVE MP4` grava:

```text
pilot_sessions/S01_20260613/context_replay.mp4
```

Para escolher outro caminho:

```bash
python -m datacol.context_replay \
  pilot_sessions/S01_20260613 \
  --output-video datasets/v1/S01_context_audit.mp4
```

Para gerar o vídeo sem abrir a janela:

```bash
python -m datacol.context_replay \
  pilot_sessions/S01_20260613 \
  --output-video datasets/v1/S01_context_audit.mp4 \
  --export-only
```

Quando o `ffmpeg` está instalado, a exportação usa H.264 (`libx264`) com
`CRF 18`, preservando melhor os textos, linhas do esqueleto e detalhes do
vídeo. Se ele não estiver disponível, a ferramenta usa `mp4v` como fallback.

## Pontos esperados na R01

```text
quadro 0:    [1, 0, 0, 0, 0, 0, 0]
quadro 1753: evento begin_four_tubes
quadro 2141: [1, 0, 0, 0, 1, 1, 0]
quadro 2347: [1, 0, 0, 0, 1, 1, 1]
```

Durante um gesto, o vetor ainda não contém o efeito daquele gesto. Por
exemplo, a quarta roda continua mostrando `0.75`; o valor passa a `1.0` no
primeiro quadro posterior ao fim desse intervalo.

Esta ferramenta audita o estado lógico e não corrige diferenças de tempo entre
FPS nominal e efetivo.
