# Estado do projeto em 13 de junho de 2026

Este documento encerra a sessão de desenvolvimento e registra o ponto exato
para retomada. O código funcional não foi alterado nesta revisão.

## Resumo

As OS-1 a OS-4 do `ROADMAP_IC.pdf` estão implementadas:

| OS | Componente | Estado |
|---|---|---|
| OS-1 | `capture_session.py` | implementada e testada |
| OS-2 | `annotate_pkl.py` | implementada e testada |
| OS-3 | `plan_sim.py` | implementada e testada |
| OS-4 | `build_json.py` | implementada e testada |

A suíte completa possui `52` testes aprovados.

A sessão `pilot_sessions/S01_20260613` validou tecnicamente:

- igualdade de `2348` quadros entre JSONL, PKL e vídeo;
- timestamps crescentes;
- `fps_effective = 18.737`;
- replay e anotação sincronizados;
- nenhuma pose ausente;
- geração de `2001` janelas;
- contexto final 7D `[1, 0, 0, 0, 1, 1, 1]`;
- auditoria gráfica e exportação do replay com contexto.

Essa sessão reproduziu gestos sem a proxy física. Ela valida o software, mas
não constitui dado experimental nem substitui o piloto real previsto no Q2.

## Alinhamento com o ROADMAP

Atendido:

- logger autônomo sem ROS, OAK-D ou PyTorch;
- landmarks MediaPipe `33 x 4` e juntas `15 x 3`;
- `frame_idx` comum aos artefatos;
- schemas de metadados, quadros, anotações e eventos;
- `ignore` explícito e excluído do dataset;
- PlanGraph offline e vetores 7D/10D;
- contexto amostrado no quadro inicial da janela;
- split integral por sessão;
- relatório de classes;
- equivalência do estágio I com `--stageI_done`.

Ainda pendente no ROADMAP:

- gabarito flat-pack físico, fotografado e documentado;
- piloto real R01 e R02 ponta a ponta;
- definição dos roteiros R03-R0n;
- coleta de 8 a 12 sessões reais;
- pelo menos 120 janelas por classe;
- no mínimo duas sessões reservadas integralmente para teste;
- OS-5, OS-6 e OS-7.

## Incompatibilidades críticas

### 1. Tempo representado por uma janela

O ROADMAP fundamenta o protocolo em aproximadamente `8 fps`. Na S01, a
captura efetiva foi `18.737 fps`; cinco quadros cobriram em média apenas
`0.213 s`.

Antes da coleta definitiva, deve ser definida uma política temporal única:

- limitar a captura;
- reamostrar offline usando `t_mono`; ou
- manter a frequência atual e justificar a mudança experimental.

Treinamento e inferência online devem usar a mesma política.

### 2. Pré-processamento numérico

O `Dataset.py` original aplica por janela:

```text
normalização min-max -> rotação por quaternion -> ajuste mínimo de Z
```

O logger armazena coordenadas MediaPipe normalizadas e registra o quaternion
nos metadados, mas não aplica essa transformação. O `build_json.py` atualmente
materializa as coordenadas armazenadas.

É necessário centralizar o pré-processamento em uma função compartilhada pelo
treinamento e pela inferência e testar sua equivalência com o pipeline legado.

### 3. Formato do `skeleton.pkl`

O novo `skeleton.pkl` contém um `numpy.ndarray` com shape `(N, 15, 3)`. O
`Dataset.py` original, no caminho PKL, espera objetos com atributo
`.landmarks`.

O JSON consolidado preserva os cut points legados, mas a leitura direta do
novo PKL pelo loader original não é compatível. A OS-5 deverá usar as janelas
materializadas ou fornecer um adaptador explícito.

### 4. Quadros sem pose

A OS-1 representa ausência de pose com zeros. A S01 não possui nenhum quadro
assim, porém a OS-4 ainda não rejeita automaticamente janelas que contenham
esse sentinela.

O próximo passo deve excluir essas janelas e registrar sua quantidade no
relatório de qualidade.

### 5. Retrocompatibilidade de `context_dim=0`

O modo `context_dim=0` emite vetor vazio, mas ainda não existe teste ponta a
ponta comprovando igualdade numérica das entradas e previsões com o
`Dataset.py` e o checkpoint original.

Esse teste é critério de aceite futuro das OS-5 e OS-6.

### 6. FPS do vídeo gravado

O vídeo da OS-1 usa o FPS nominal no contêiner, embora a captura real possa
ser mais lenta. A correspondência por `frame_idx` permanece correta, mas a
reprodução não preserva necessariamente a duração real.

Auditorias temporais devem usar `t_mono`. Uma futura melhoria pode gerar uma
versão do vídeo com tempo real sem alterar os dados brutos.

## Melhorias propostas para a próxima sessão

Prioridade alta, antes de coletar em volume:

1. implementar amostragem temporal baseada em `t_mono`;
2. criar módulo único de pré-processamento legado;
3. excluir janelas com pose ausente e ampliar o relatório de qualidade;
4. testar o loader que será usado pela OS-5 com o dataset gerado.

Prioridade de protocolo:

5. criar checklist automático por sessão;
6. documentar e fotografar o layout `v1`;
7. formalizar R01-R0n, incluindo ritmo lento, normal e rápido;
8. executar piloto físico R01 e R02 antes da coleta definitiva.

Melhorias secundárias:

9. gerar vídeo de auditoria respeitando `t_mono`;
10. adicionar relatório agregado entre sessões;
11. registrar versão do código e hash do protocolo no dataset congelado.

## Ordem de retomada

Na próxima sessão:

1. revisar este documento e decidir a política temporal;
2. implementar e testar os itens críticos 1 a 4;
3. atualizar o protocolo caso o significado temporal da janela mude;
4. montar e registrar o layout físico;
5. executar somente o piloto real;
6. autorizar a coleta em volume apenas após revisar o piloto.

Não use `pilot_dataset.json` para treinamento experimental. Ele é um artefato
de teste do pipeline.
