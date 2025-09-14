#!/usr/bin/env python3
"""
mediapipe_video_test.py

Uso:
    python mediapipe_video_test.py --input video.mp4 --output_annotated out.mp4 --output_csv landmarks.csv [--use_pose]

Descrição:
    Processa o vídeo usando MediaPipe Hands (opcional Pose), desenha landmarks e salva:
    - vídeo anotado com desenhos
    - CSV com landmarks por frame

CSV format:
    frame, hand0_x_0, hand0_y_0, hand0_z_0, ..., hand0_x_20, hand0_y_20, hand0_z_20,
           hand1_x_0, ... (se mão ausente -> NaN)
    (se --use_pose incluído, as colunas de pose aparecem depois: pose_x_0, pose_y_0, pose_z_0, ...)
"""
import argparse
import math
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def build_header(use_pose=False, max_hands=2):
    header = ["frame"]
    # Hands: each hand has 21 landmarks, each landmark has x,y,z
    for h in range(max_hands):
        for i in range(21):
            header += [f"hand{h}_x_{i}", f"hand{h}_y_{i}", f"hand{h}_z_{i}"]
        # Optional handedness score/confidence
        header += [f"hand{h}_presence"]  # 1 if this hand detected, 0 otherwise
    if use_pose:
        # Pose: 33 landmarks in Mediapipe Pose
        for i in range(33):
            header += [f"pose_x_{i}", f"pose_y_{i}", f"pose_z_{i}"]
    return header

def extract_hand_landmarks(results_hands, max_hands=2):
    """
    returns list of shape (max_hands, 21*3) flattened; if fewer hands detected fills with np.nan
    Also returns presence flags list of length max_hands
    The detected order corresponds to results_hands.multi_handedness (if available)
    """
    out = []
    presence = []
    if results_hands.multi_hand_landmarks:
        # To keep stable order, use the handedness label index order
        # Build a list of (handedness_index, landmarks) pairs
        landmarks_list = results_hands.multi_hand_landmarks
        handedness_list = results_hands.multi_handedness  # list of Classification
        # We'll combine them in the same order returned
        for lm in landmarks_list:
            flat = []
            for p in lm.landmark:
                flat.extend([p.x, p.y, p.z])
            out.append(flat)
            presence.append(1)
    # pad to max_hands
    while len(out) < max_hands:
        out.append([math.nan] * (21 * 3))
        presence.append(0)
    # if more than max_hands, truncate
    out = out[:max_hands]
    presence = presence[:max_hands]
    return out, presence

def extract_pose_landmarks(results_pose):
    if results_pose.pose_landmarks:
        flat = []
        for p in results_pose.pose_landmarks.landmark:
            flat.extend([p.x, p.y, p.z])
        return flat
    else:
        return [math.nan] * (33 * 3)

def main(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(args.output_annotated, fourcc, fps, (width, height))

    header = build_header(use_pose=args.use_pose, max_hands=args.max_hands)
    rows = []

    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    pose = None
    if args.use_pose:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence
        )

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing frames")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False

            results_hands = hands.process(img_rgb)
            results_pose = None
            if pose:
                results_pose = pose.process(img_rgb)

            img_rgb.flags.writeable = True
            annotated = frame.copy()

            # Draw hands
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,128,255), thickness=2)
                    )

            # Draw pose
            if results_pose and results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,128,0), thickness=2)
                )

            # extract landmarks into row
            row = [frame_idx]
            hand_flat_list, hand_presence = extract_hand_landmarks(results_hands, max_hands=args.max_hands)
            for h_flat in hand_flat_list:
                row.extend(h_flat)
            for pres in hand_presence:
                row.append(pres)
            if args.use_pose:
                row.extend(extract_pose_landmarks(results_pose))

            rows.append(row)

            # write annotated frame to output video
            out_vid.write(annotated)

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        out_vid.release()
        hands.close()
        if pose:
            pose.close()

    # Save CSV
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(args.output_csv, index=False)
    print(f"Anotated video salvo em: {args.output_annotated}")
    print(f"Landmarks CSV salvo em: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa vídeo com MediaPipe Hands (e opcionalmente Pose).")
    parser.add_argument("--input", required=True, help="Caminho para o vídeo de entrada (ex: video.mp4)")
    parser.add_argument("--output_annotated", required=True, help="Caminho para salvar o vídeo anotado (ex: annotated.mp4)")
    parser.add_argument("--output_csv", required=True, help="Caminho para salvar o CSV de landmarks (ex: landmarks.csv)")
    parser.add_argument("--use_pose", action="store_true", help="Incluir landmarks de Pose (útil para contexto corporal)")
    parser.add_argument("--max_hands", type=int, default=2, help="Número máximo de mãos para capturar (default 2)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
