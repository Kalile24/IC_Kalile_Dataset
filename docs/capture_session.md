# Guia do `capture_session.py`

O `capture_session.py` implementa a OS-1 do roadmap: captura webcam e
MediaPipe Pose, mantendo a mesma chave `frame_idx` em:

- `meta.json`;
- `frames.jsonl`;
- `skeleton.pkl`;
- `video.mp4`.

O logger nao usa ROS, OAK-D ou PyTorch e nao aplica EMA, suavizacao do
MediaPipe, regras OOD ou filtros de inferencia.

## Preparacao

O projeto requer Python 3.9 e MediaPipe Solutions Pose:

```bash
cd /home/marcos-kalile/hrc-data-collection
source .venv/bin/activate
python -m pip install --upgrade setuptools wheel
python -m pip install --no-build-isolation -e ".[dev]"
```

Confirme o ambiente:

```bash
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"
python -m datacol.capture_session --help
```

A primeira linha deve terminar em `True`.

## Descobrir cameras

Liste somente indices que abrem e entregam um quadro:

```bash
python -m datacol.capture_session --list-cameras
```

Exemplo:

```text
index=2 resolution=1280x720
```

Por padrao, sao testados os indices de 0 a 9. Para ampliar:

```bash
python -m datacol.capture_session --list-cameras --camera-scan-limit 20
```

Durante a captura, `--camera-index` e opcional. Quando omitido, a primeira
camera legivel e selecionada automaticamente.

## Identificador da sessao

Consulte o proximo ID sem iniciar uma captura:

```bash
python -m datacol.capture_session --suggest-session-id
```

O formato e `SNN_YYYYMMDD`. O numero e sempre superior ao maior numero de
sessao existente, inclusive se a sessao anterior tiver outra data. Isso evita
reutilizacao acidental de IDs.

O argumento posicional `session_id` e opcional. Quando omitido, o logger exibe
e usa automaticamente a sugestao.

## Iniciar uma captura

Forma recomendada, com ID e camera automaticos:

```bash
python -m datacol.capture_session \
  --participant P01 \
  --script-id R01 \
  --camera-model "Intelbras WCI 1080p" \
  --camera-distance-m 2.2 \
  --width 1280 \
  --height 720 \
  --fps 30
```

Para controlar explicitamente ID e camera:

```bash
python -m datacol.capture_session S07_20260613 \
  --participant P01 \
  --script-id R01 \
  --camera-model "Intelbras WCI 1080p" \
  --camera-distance-m 2.2 \
  --camera-index 2 \
  --width 1280 \
  --height 720 \
  --fps 30
```

Opcoes comuns:

- `--max-frames 150`: encerra automaticamente apos 150 quadros;
- `--no-preview`: captura sem abrir janela;
- `--autofocus locked_v4l2`: registra a configuracao de foco;
- `--camera-height chest`: registra a altura da camera;
- `--zone-layout-version v1`: registra a versao do layout;
- `--output-root sessions`: altera o diretorio de sessoes;
- `--quaternion-world W X Y Z`: altera o quaternion registrado.

## Janela de captura

O HUD mostra:

- `REC` e o numero de quadros gravados;
- tempo monotonicamente decorrido;
- FPS observado durante a captura;
- `POSE OK` em verde ou `POSE MISSING` em vermelho;
- lembrete das teclas de encerramento.

Pressione `q` ou `Esc` para finalizar. `Ctrl+C` tambem encerra preservando os
quadros ja capturados. O HUD existe apenas na visualizacao: `video.mp4` recebe
o quadro original, sem texto sobreposto.

Se o MediaPipe nao detectar pose, o quadro ainda e gravado para preservar o
sincronismo. Nesse caso:

- `landmarks_raw` recebe 33 entradas `[0, 0, 0, 0]`;
- `joints15` recebe 15 entradas `[0, 0, 0]`.

Nenhuma pose anterior e repetida ou imputada.

## Saida

Uma sessao finalizada possui:

```text
sessions/S07_20260613/
├── meta.json
├── frames.jsonl
├── skeleton.pkl
└── video.mp4
```

`meta.json` registra participante, roteiro, data com fuso, camera, resolucao
real, FPS nominal e efetivo, geometria, versoes e quaternion.

Cada linha de `frames.jsonl` contem:

```json
{
  "frame_idx": 0,
  "t_mono": 0.0,
  "landmarks_raw": [[0.0, 0.0, 0.0, 0.0]],
  "joints15": [[0.0, 0.0, 0.0]]
}
```

Os arrays reais possuem respectivamente 33 e 15 itens.

`skeleton.pkl` contem um `numpy.ndarray` `float32` com shape `(N, 15, 3)`.
`video.mp4` contem os mesmos `N` quadros e na mesma ordem.

## Validar uma sessao

A captura executa validacao automaticamente ao finalizar. Para repetir:

```bash
python -m datacol.capture_session \
  --validate-session sessions/S07_20260613
```

O comando verifica:

- presenca dos quatro arquivos;
- estrutura e campos de `meta.json`;
- shapes `(33, 4)` e `(15, 3)` em cada linha JSONL;
- `frame_idx` zero-based e contiguo;
- timestamps finitos e estritamente crescentes;
- shape `(N, 15, 3)` do PKL;
- igualdade entre `joints15` do JSONL e o PKL;
- igualdade de `N` entre JSONL, PKL e video;
- preenchimento de `fps_effective`.

O codigo de saida e `0` para sessao valida e `1` para sessao invalida.

## Protecao dos dados

Um diretorio existente nunca e sobrescrito. Use outro ID para uma nova
captura.

Se a inicializacao falhar antes de gravar o primeiro quadro, o logger remove
somente o diretorio vazio criado por aquela tentativa. Se ao menos um quadro
tiver sido gravado, os arquivos parciais sao preservados para auditoria e
recuperacao manual.

## Testes

Execute apenas a OS-1:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_integrity.py
```

O bloqueio de plugins evita que plugins pytest do ROS global contaminem este
repositorio independente.

## Limitações conhecidas

O vídeo é gravado com o FPS nominal informado à câmera, enquanto
`fps_effective` e `t_mono` registram o ritmo real. A sincronização por
`frame_idx` permanece correta, mas a duração da reprodução pode diferir do
tempo real.

As juntas armazenadas são coordenadas normalizadas do MediaPipe. O quaternion
é registrado em `meta.json`, mas o logger não aplica rotação nem normalização
de treinamento. Essas decisões pertencem ao pré-processamento offline.

Consulte
[`session_status_2026-06-13.md`](session_status_2026-06-13.md) antes da
próxima coleta experimental.
