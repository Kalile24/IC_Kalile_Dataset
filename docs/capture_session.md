# Guia do `capture_session.py`

O `capture_session.py` implementa a OS-1 do roadmap: captura webcam e
MediaPipe Pose, mantendo a mesma chave `frame_idx` em:

- `meta.json`;
- `frames.jsonl`;
- `skeleton.pkl`;
- `video.mp4`.

O logger não usa ROS, OAK-D ou PyTorch e não aplica EMA, suavização do
MediaPipe, regras OOD ou filtros de inferência.

## Preparação

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

## Descobrir câmeras

Liste somente índices que abrem e entregam um quadro:

```bash
python -m datacol.capture_session --list-cameras
```

Exemplo:

```text
index=2 resolution=1280x720
```

Por padrão, são testados os índices de 0 a 9. Para ampliar:

```bash
python -m datacol.capture_session --list-cameras --camera-scan-limit 20
```

Durante a captura, `--camera-index` e opcional. Quando omitido, a primeira
câmera legível e selecionada automaticamente.

## Identificador da sessão

Consulte o próximo ID sem iniciar uma captura:

```bash
python -m datacol.capture_session --suggest-session-id
```

O formato e `SNN_YYYYMMDD`. O número e sempre superior ao maior número de
sessão existente, inclusive se a sessão anterior tiver outra data. Isso evita
reutilização acidental de IDs.

O argumento posicional `session_id` e opcional. Quando omitido, o logger exibe
e usa automaticamente a sugestão.

## Iniciar uma captura

Forma recomendada, com ID e câmera automáticos:

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

Para controlar explicitamente ID e câmera:

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

Opções comuns:

- `--max-frames 150`: encerra automaticamente após 150 quadros;
- `--no-preview`: captura sem abrir janela;
- `--autofocus locked_v4l2`: registra a configuração de foco;
- `--camera-height chest`: registra a altura da câmera;
- `--zone-layout-version v1`: registra a versão do layout;
- `--output-root sessions`: altera o diretório de sessões;
- `--quaternion-world W X Y Z`: altera o quaternion registrado.

## Janela de captura

O HUD mostra:

- `REC` e o número de quadros gravados;
- tempo monotonicamente decorrido;
- FPS observado durante a captura;
- `POSE OK` em verde ou `POSE MISSING` em vermelho;
- lembrete das teclas de encerramento.

Pressione `q` ou `Esc` para finalizar. `Ctrl+C` também encerra preservando os
quadros já capturados. O HUD existe apenas na visualização: `video.mp4` recebe
o quadro original, sem texto sobreposto.

Se o MediaPipe não detectar pose, o quadro ainda e gravado para preservar o
sincronismo. Nesse caso:

- `landmarks_raw` recebe 33 entradas `[0, 0, 0, 0]`;
- `joints15` recebe 15 entradas `[0, 0, 0]`.

Nenhuma pose anterior e repetida ou imputada.

## Saída

Uma sessão finalizada possui:

```text
sessions/S07_20260613/
├── meta.json
├── frames.jsonl
├── skeleton.pkl
└── video.mp4
```

`meta.json` registra participante, roteiro, data com fuso, câmera, resolução
real, FPS nominal e efetivo, geometria, versões e quaternion.

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

## Validar uma sessão

A captura executa validação automaticamente ao finalizar. Para repetir:

```bash
python -m datacol.capture_session \
  --validate-session sessions/S07_20260613
```

O comando verifica:

- presença dos quatro arquivos;
- estrutura e campos de `meta.json`;
- shapes `(33, 4)` e `(15, 3)` em cada linha JSONL;
- `frame_idx` zero-based e contiguo;
- timestamps finitos e estritamente crescentes;
- shape `(N, 15, 3)` do PKL;
- igualdade entre `joints15` do JSONL e o PKL;
- igualdade de `N` entre JSONL, PKL e video;
- preenchimento de `fps_effective`.

O código de saída é `0` para sessão válida e `1` para sessão inválida.

## Proteção dos dados

Um diretório existente nunca e sobrescrito. Use outro ID para uma nova
captura.

Se a inicialização falhar antes de gravar o primeiro quadro, o logger remove
somente o diretório vazio criado por aquela tentativa. Se ao menos um quadro
tiver sido gravado, os arquivos parciais são preservados para auditoria e
recuperação manual.

## Testes

Execute apenas a OS-1:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_integrity.py
```

O bloqueio de plugins evita que plugins pytest do ROS global contaminem este
repositório independente.

## Limitações conhecidas

O vídeo é gravado com o FPS nominal informado à câmera, enquanto
`fps_effective` e `t_mono` registram o ritmo real. A sincronização por
`frame_idx` permanece correta, mas a duração da reprodução pode diferir do
tempo real.

As juntas armazenadas são coordenadas normalizadas do MediaPipe. O quaternion
é registrado em `meta.json`, mas o logger não aplica rotação nem normalização
de treinamento. Essas decisões pertencem ao pré-processamento offline,
aplicado em `hrc-finetune` no momento do fine-tuning.
