# dataset-libras

Dataset de sinais em **Libras** (Língua Brasileira de Sinais) para treinamento de modelos de
Machine Learning, com pipeline de extração de **landmarks de mãos** em Python + MediaPipe.

## O que tem aqui

| Caminho | Conteúdo |
|---|---|
| `workspace/images/train/` | vídeos originais |
| `workspace/annotated-videos/<sinal>/` | os mesmos vídeos com o esqueleto da mão sobreposto (259) |
| `workspace/landmarks/<sinal>/` | CSV de coordenadas por gravação (259 arquivos, 422 colunas) |
| `metadados/<sinal>/` | JSON de metadados por gravação (259 arquivos) |
| `scripts/mediapipe_hand_tracker.py` | pipeline de extração |

## Números

- **10 sinais**: Algoritmo, Amor, Azul, Bom-dia, Camarão, Cientista, Computador, IA, J, Melão
- **9 sinalizadores** × **3 repetições** por sinal
- **259 gravações** processadas (de 270 previstas)
- **49.506 frames**, dos quais **49.281 (99,5%)** com mãos detectadas
- **422 atributos por frame**: `frame`, `timestamp` + 21 landmarks × 2 mãos × (x, y, z, visibility, presence)
- Resoluções 1920x1080 e 3840x2160 · FPS de 29 a 59

## Pipeline

`scripts/mediapipe_hand_tracker.py` (Python, MediaPipe Hands + OpenCV) faz, por vídeo:

1. abre o vídeo e processa frame a frame;
2. detecta e rastreia até 2 mãos (21 landmarks cada);
3. grava o vídeo anotado com o esqueleto sobreposto;
4. exporta o CSV de coordenadas;
5. grava o JSON de metadados com resolução, FPS, total de frames, frames com mãos e a
   configuração exata do MediaPipe usada;
6. roda uma rotina de auditoria de qualidade sobre o CSV gerado.

A configuração vive numa `dataclass` (confiança de detecção 0.7, confiança de rastreio 0.5,
`max_num_hands=2`, `model_complexity=1`), então cada JSON registra os parâmetros que produziram
aquela amostra: **o dataset é reprodutível a partir do próprio metadado**.

## Estrutura dos metadados

```json
{
  "video_original": "...",
  "total_frames": 112,
  "frames_com_maos": 111,
  "fps": 29,
  "resolucao": "1920x1080",
  "configuracoes_mediapipe": { "min_detection_confidence": 0.7, "...": "..." },
  "colunas_extraidas": ["frame", "timestamp", "Right_0_WRIST_x", "..."]
}
```

## Como reproduzir

```bash
pip install opencv-python mediapipe pandas numpy tqdm
python scripts/mediapipe_hand_tracker.py
```

## Contexto

Projeto de equipe no IFCE — Campus Aracati (Ago–Dez 2025). A coleta foi desenhada junto à
**comunidade surda**, para garantir representatividade dos sinais e das pessoas sinalizantes.

## Problemas conhecidos

- Nomes de sinalizador e de sinal aparecem em mais de uma grafia (`Neuriania` / `Neurinia`,
  `Camarão` / `Camarao`, `Melão` / `Melao`). Normalizar antes de usar como rótulo de treino.
- 11 das 270 gravações previstas não foram coletadas — o conjunto está desbalanceado entre
  sinalizadores.
- Não há split de treino/validação/teste definido: o dataset é a matéria-prima, não o
  experimento.

## Licença

MIT — ver [LICENSE](LICENSE).
