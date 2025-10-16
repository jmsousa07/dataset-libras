"""
Sistema Profissional de Extração de Pontos de Mãos para Dataset de Libras
Autor: Sistema desenvolvido para processamento de sinais de Libras
Versão: 2.0
Data: 2024
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging profissional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processamento_libras.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConfiguracaoProcessamento:
    """Classe para armazenar configurações do processamento"""
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2
    model_complexity: int = 1
    static_image_mode: bool = False
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    
class ProcessadorVideoLibras:
    """
    Classe principal para processar vídeos de Libras e extrair pontos das mãos.
    Utiliza MediaPipe para detecção e rastreamento de mãos em tempo real.
    """
    
    def __init__(self, config: Optional[ConfiguracaoProcessamento] = None):
        """
        Inicializa o processador com configurações específicas.
        
        Args:
            config: Configurações do MediaPipe. Se None, usa configurações padrão.
        """
        self.config = config or ConfiguracaoProcessamento()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializar detector de mãos
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.config.static_image_mode,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            model_complexity=self.config.model_complexity
        )
        
        # Nomes dos pontos de referência das mãos
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        logger.info("ProcessadorVideoLibras inicializado com sucesso")
        logger.info(f"Configurações: {asdict(self.config)}")
    
    def processar_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """
        Processa um único frame e extrai coordenadas das mãos.
        
        Args:
            frame: Frame do vídeo como array numpy
            
        Returns:
            Tupla contendo:
                - Dicionário com coordenadas normalizadas (ou None se não detectar mãos)
                - Frame anotado com desenho dos pontos
        """
        # Converter BGR para RGB (MediaPipe usa RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Detectar mãos
        results = self.hands.process(frame_rgb)
        
        # Preparar frame para anotação
        frame_rgb.flags.writeable = True
        frame_anotado = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        coordenadas = {}
        
        if results.multi_hand_landmarks:
            for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Identificar se é mão esquerda ou direita
                label = handedness.classification[0].label
                hand_prefix = f"{label}_{hand_idx}"
                
                # Extrair e normalizar coordenadas
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmark_name = self.landmark_names[idx]
                    
                    # Coordenadas normalizadas (0-1)
                    coordenadas[f"{hand_prefix}_{landmark_name}_x"] = landmark.x
                    coordenadas[f"{hand_prefix}_{landmark_name}_y"] = landmark.y
                    coordenadas[f"{hand_prefix}_{landmark_name}_z"] = landmark.z
                    
                    # Adicionar visibilidade e presença
                    coordenadas[f"{hand_prefix}_{landmark_name}_visibility"] = landmark.visibility
                    coordenadas[f"{hand_prefix}_{landmark_name}_presence"] = landmark.presence if hasattr(landmark, 'presence') else 1.0
                
                # Desenhar pontos e conexões no frame
                self.mp_drawing.draw_landmarks(
                    frame_anotado,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return coordenadas if coordenadas else None, frame_anotado
    
    def processar_video(self, caminho_video: str, caminho_saida: str, 
                       salvar_video_anotado: bool = True) -> pd.DataFrame:
        """
        Processa um vídeo completo e salva os resultados.
        
        Args:
            caminho_video: Caminho para o vídeo de entrada
            caminho_saida: Caminho base para salvar os arquivos de saída
            salvar_video_anotado: Se True, salva o vídeo com anotações
            
        Returns:
            DataFrame com todas as coordenadas extraídas
        """
        logger.info(f"Iniciando processamento do vídeo: {caminho_video}")
        
        # Verificar se o vídeo existe
        if not os.path.exists(caminho_video):
            logger.error(f"Vídeo não encontrado: {caminho_video}")
            raise FileNotFoundError(f"Vídeo não encontrado: {caminho_video}")
        
        # Abrir vídeo
        cap = cv2.VideoCapture(caminho_video)
        
        # Obter propriedades do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Propriedades do vídeo - FPS: {fps}, Resolução: {width}x{height}, Total de frames: {total_frames}")
        
        # Preparar writer para vídeo anotado
        video_writer = None
        if salvar_video_anotado:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_saida = f"{caminho_saida}_anotado.mp4"
            video_writer = cv2.VideoWriter(video_saida, fourcc, fps, (width, height))
            logger.info(f"Vídeo anotado será salvo em: {video_saida}")
        
        # Lista para armazenar dados de todos os frames
        dados_frames = []
        frames_com_maos = 0
        
        # Processar cada frame com barra de progresso
        with tqdm(total=total_frames, desc="Processando frames") as pbar:
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Processar frame
                coordenadas, frame_anotado = self.processar_frame(frame)
                
                # Preparar dados do frame
                dados_frame = {'frame': frame_num, 'timestamp': frame_num / fps}
                
                if coordenadas:
                    dados_frame.update(coordenadas)
                    frames_com_maos += 1
                
                dados_frames.append(dados_frame)
                
                # Salvar frame anotado
                if salvar_video_anotado and video_writer:
                    video_writer.write(frame_anotado)
                
                frame_num += 1
                pbar.update(1)
        
        # Liberar recursos
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Criar DataFrame
        df = pd.DataFrame(dados_frames)
        
        # Preencher valores NaN com 0 para frames sem detecção
        df = df.fillna(0)
        
        # Salvar CSV
        csv_saida = f"{caminho_saida}_coordenadas.csv"
        df.to_csv(csv_saida, index=False, float_format='%.6f')
        logger.info(f"CSV salvo em: {csv_saida}")
        
        # Estatísticas do processamento
        logger.info(f"Processamento concluído!")
        logger.info(f"Total de frames processados: {frame_num}")
        logger.info(f"Frames com mãos detectadas: {frames_com_maos} ({frames_com_maos/frame_num*100:.1f}%)")
        logger.info(f"Dimensões do DataFrame: {df.shape}")
        
        # Salvar metadados
        metadados = {
            'video_original': caminho_video,
            'total_frames': frame_num,
            'frames_com_maos': frames_com_maos,
            'fps': fps,
            'resolucao': f"{width}x{height}",
            'configuracoes_mediapipe': asdict(self.config),
            'colunas_extraidas': list(df.columns),
            'taxa_deteccao': f"{frames_com_maos/frame_num*100:.2f}%"
        }
        
        metadados_saida = f"{caminho_saida}_metadados.json"
        with open(metadados_saida, 'w', encoding='utf-8') as f:
            json.dump(metadados, f, indent=4, ensure_ascii=False)
        logger.info(f"Metadados salvos em: {metadados_saida}")
        
        return df
    
    def processar_lote(self, pasta_videos: str, pasta_saida: str, 
                      extensoes: List[str] = ['.mp4']) -> Dict:
        """
        Processa múltiplos vídeos de uma pasta.
        
        Args:
            pasta_videos: Pasta contendo os vídeos
            pasta_saida: Pasta para salvar os resultados
            extensoes: Lista de extensões de vídeo aceitas
            
        Returns:
            Dicionário com estatísticas do processamento
        """
        logger.info(f"Iniciando processamento em lote da pasta: {pasta_videos}")
        
        # Criar pasta de saída se não existir
        Path(pasta_saida).mkdir(parents=True, exist_ok=True)
        
        # Listar vídeos
        videos = []
        for ext in extensoes:
            videos.extend(Path(pasta_videos).glob(f'*{ext}'))
            videos.extend(Path(pasta_videos).glob(f'*{ext.upper()}'))
        
        logger.info(f"Encontrados {len(videos)} vídeos para processar")
        
        resultados = {
            'videos_processados': 0,
            'videos_com_erro': 0,
            'erros': [],
            'estatisticas': []
        }
        
        for video_path in videos:
            try:
                logger.info(f"\nProcessando: {video_path.name}")
                
                # Definir caminhos de saída
                nome_base = video_path.stem
                caminho_saida = os.path.join(pasta_saida, nome_base)
                
                # Processar vídeo
                df = self.processar_video(
                    str(video_path),
                    caminho_saida,
                    salvar_video_anotado=True
                )
                
                resultados['videos_processados'] += 1
                resultados['estatisticas'].append({
                    'video': video_path.name,
                    'frames_totais': len(df),
                    'colunas': df.shape[1]
                })
                
            except Exception as e:
                logger.error(f"Erro ao processar {video_path.name}: {str(e)}")
                resultados['videos_com_erro'] += 1
                resultados['erros'].append({
                    'video': video_path.name,
                    'erro': str(e)
                })
        
        # Salvar resumo do processamento
        resumo_path = os.path.join(pasta_saida, 'resumo_processamento.json')
        with open(resumo_path, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSAMENTO CONCLUÍDO")
        logger.info(f"Vídeos processados com sucesso: {resultados['videos_processados']}")
        logger.info(f"Vídeos com erro: {resultados['videos_com_erro']}")
        logger.info(f"Resumo salvo em: {resumo_path}")
        logger.info(f"{'='*50}")
        
        return resultados
    
    def __del__(self):
        """Limpar recursos ao destruir o objeto"""
        if hasattr(self, 'hands'):
            self.hands.close()

def analisar_qualidade_dados(csv_path: str) -> Dict:
    """
    Analisa a qualidade dos dados extraídos.
    
    Args:
        csv_path: Caminho para o arquivo CSV
        
    Returns:
        Dicionário com métricas de qualidade
    """
    df = pd.read_csv(csv_path)
    
    analise = {
        'total_frames': len(df),
        'frames_com_dados': (df.iloc[:, 2:] != 0).any(axis=1).sum(),
        'taxa_deteccao': 0,
        'colunas_totais': df.shape[1],
        'valores_nulos': df.isnull().sum().sum(),
        'estatisticas_coordenadas': {}
    }
    
    analise['taxa_deteccao'] = (analise['frames_com_dados'] / analise['total_frames']) * 100
    
    # Estatísticas para coordenadas x e y
    coord_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]
    if coord_cols:
        analise['estatisticas_coordenadas'] = {
            'media': df[coord_cols].mean().mean(),
            'desvio_padrao': df[coord_cols].std().mean(),
            'min': df[coord_cols].min().min(),
            'max': df[coord_cols].max().max()
        }
    
    return analise

# ================== SCRIPT PRINCIPAL ==================

def main():
    """
    Função principal para executar o processamento.
    """
    print("\n" + "="*60)
    print("   SISTEMA DE EXTRAÇÃO DE PONTOS DE LIBRAS - v2.0")
    print("="*60 + "\n")
    
    # Configurações personalizadas (ajuste conforme necessário)
    config = ConfiguracaoProcessamento(
        min_detection_confidence=0.7,  # Confiança mínima para detecção
        min_tracking_confidence=0.5,   # Confiança mínima para rastreamento
        max_num_hands=2,               # Máximo de mãos a detectar
        model_complexity=1              # Complexidade do modelo (0, 1 ou 2)
    )
    
    # Criar processador
    processador = ProcessadorVideoLibras(config)
    
    # # Exemplo de uso individual
    # print("\n### EXEMPLO DE USO INDIVIDUAL ###")
    # print("Para processar um único vídeo:")
    # print("----------------------------------------")
    
    # # ALTERE ESTES CAMINHOS CONFORME SUA ESTRUTURA
    # video_individual = "caminho/para/seu/video.mp4"
    # saida_individual = "caminho/para/saida/nome_do_arquivo"
    
    # print(f"video_path = '{video_individual}'")
    # print(f"output_path = '{saida_individual}'")
    # print("\n# Processar vídeo")
    # print("df = processador.processar_video(video_path, output_path, salvar_video_anotado=True)")
    
    # Exemplo de processamento em lote
    print("\n### EXEMPLO DE PROCESSAMENTO EM LOTE ###")
    print("Para processar múltiplos vídeos:")
    print("----------------------------------------")
    
    # ALTERE ESTES CAMINHOS CONFORME SUA ESTRUTURA
    pasta_videos = "C:/Users/LEMA/Documents/dataset-libras/workspace/images/train"
    pasta_saida = "C:/Users/LEMA/Documents/dataset-libras/results"
    resultados = processador.processar_lote(pasta_videos, pasta_saida)
    
    print(f"pasta_videos = '{pasta_videos}'")
    print(f"pasta_saida = '{pasta_saida}'")
    print("\n# Processar todos os vídeos da pasta")
    print("resultados = processador.processar_lote(pasta_videos, pasta_saida)")
    
    print("\n" + "="*60)
    print("Configure os caminhos acima e execute o processamento!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()