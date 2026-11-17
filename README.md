# dataset-libras

Dataset de sinais em **Libras** (Língua Brasileira de Sinais) para treinamento de modelos de
Machine Learning, com pipeline de extração de **landmarks de mãos** em Python + MediaPipe.

> Trabalho de iniciação científica aprovado para apresentação no **XV CONNEPI** (Congresso Norte
> Nordeste de Pesquisa e Inovação), IFCE, Fortaleza, 9–11 de dezembro de 2026.
> Ver [Publicação](#publicação).

## O que tem aqui

| Caminho | Conteúdo |
|---|---|
| `workspace/images/train/` | vídeos originais |
| `workspace/annotated-videos/<sinal>/` | os mesmos vídeos com o esqueleto da mão sobreposto (259) |
| `workspace/landmarks/<sinal>/` | CSV de coordenadas por gravação (259 arquivos, 422 colunas) |
| `metadados/<sinal>/` | JSON de metadados por gravação (259 arquivos) |
| `scripts/mediapipe_hand_tracker.py` | pipeline de extração |
| `paper/` | trabalho completo submetido ao CONNEPI 2026 (PDF) |

## Números

Do trabalho aprovado:

- **10 sinais** de Libras, escolhidos por diversidade de movimento e por dois campos semânticos
  (tecnologia e cotidiano)
- **10 sinalizadores** do IFCE Campus Aracati, em três perfis de fluência: intérpretes, surdos
  nativos e aprendizes
- **3 ângulos** por sinal (frontal R1, lateral direito R2 a +45°, lateral esquerdo R3 a −45°),
  capturados simultaneamente por três iPhones (1080p30, 1080p60 e 4K60)
- **297 gravações validadas** de 300 planejadas — taxa de aprovação de 99%
- **92%** dos vídeos classificados como de qualidade boa ou excelente por dois avaliadores
  independentes (concordância κ = 0,847)
- Taxa de detecção de landmarks **> 95%**, com ~90% de detecção bimanual

Estado deste repositório:

- **259 gravações processadas** — as demais estão em processamento
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

O artigo registra as versões usadas na coleta original: MediaPipe v0.10.21, Python 3.10,
OpenCV 4.8.1, NumPy 1.24 e Pandas 2.0.

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

## Publicação

SOUSA CASTRO, J. M. de; SIEBRA, A. H. de C.; OLIVEIRA, P. A. G. de; AMARAL, D. S. do.
**Desenvolvimento de um dataset de sinais em Libras para machine learning**. In: CONGRESSO
NORTE NORDESTE DE PESQUISA E INOVAÇÃO, 15., 2026, Fortaleza. Trabalho aprovado para
apresentação. Fortaleza: IFCE, 2026.

- Arquivo: [`paper/Sousa-Castro-et-al-CONNEPI-2026.pdf`](paper/Sousa-Castro-et-al-CONNEPI-2026.pdf)
- Evento: XV CONNEPI — Centro de Eventos do Ceará, Fortaleza, 9 a 11 de dezembro de 2026
- Orientação: Profa. Dra. Denise Silva do Amaral

## Contexto

Projeto de equipe no IFCE — Campus Aracati (Ago–Dez 2025). A coleta foi desenhada junto à
**comunidade surda**, com apoio de intérpretes certificados e de uma pessoa surda na seleção dos
sinais, no protocolo de captura e na validação. Todos os participantes assinaram Termo de
Consentimento Livre e Esclarecido, apresentado em Libras aos participantes surdos.

## Problemas conhecidos

- **Cobertura parcial**: o artigo documenta 297 gravações validadas; este repositório traz 259
  processadas.
- **Divergência de vocabulário**: o artigo lista o sinal *peixe*; no repositório ele aparece como
  `Camarão`. Reconciliar antes de usar os rótulos.
- **Grafia inconsistente** de sinalizador e sinal (`Neuriania` / `Neurinia`, `Camarão` / `Camarao`,
  `Melão` / `Melao`). Normalizar antes de usar como rótulo de treino.
- **Formato de anotação**: o artigo descreve a saída de coordenadas em JSON; o pipeline deste
  repositório exporta CSV, com o JSON reservado aos metadados.
- Não há split de treino/validação/teste definido: o dataset é a matéria-prima, não o experimento.

## Licença

- **Código** (`scripts/`): Apache License 2.0 — ver [LICENSE](LICENSE).
- **Dados** (vídeos, landmarks e metadados): **CC BY 4.0**, conforme declarado no artigo.
