#!/usr/bin/env python3
"""
overlay_landmarks_from_csv.py

Uso:
python overlay_landmarks_from_csv.py --video path/to/video.mp4 --csv path/to/landmarks.csv --out out_overlay.mp4 --max_hands 2

Gera um vídeo com os landmarks desenhados usando os valores do CSV (assume coords normalizadas 0..1).
"""
import argparse
import cv2
import pandas as pd
import numpy as np
import os

def draw_landmarks_from_row(img, row, max_hands=2, circle_radius=3, color_hand0=(0,255,0), color_hand1=(0,128,255)):
    h, w = img.shape[:2]
    colors = [color_hand0, color_hand1]
    for hand in range(max_hands):
        x_cols = [c for c in row.index if c.startswith(f"hand{hand}_x_")]
        y_cols = [c for c in row.index if c.startswith(f"hand{hand}_y_")]
        pres_col = f"hand{hand}_presence"
        present = False
        if pres_col in row.index:
            try:
                present = bool(int(row[pres_col]))
            except:
                present = not all(np.isnan([row[c] for c in x_cols+y_cols]))
        if not x_cols or not y_cols:
            continue
        xs = np.array([row[c] for c in x_cols], dtype=float)
        ys = np.array([row[c] for c in y_cols], dtype=float)
        if np.all(np.isnan(xs)) or np.all(np.isnan(ys)):
            continue
        for i,(xn,yn) in enumerate(zip(xs,ys)):
            # if normalized coords (0..1) check and clamp
            if np.isnan(xn) or np.isnan(yn):
                continue
            # If values possibly >1 or <0, still convert but clamp to image
            px = int(np.clip(xn, 0.0, 1.0) * w)
            py = int(np.clip(yn, 0.0, 1.0) * h)
            cv2.circle(img, (px, py), circle_radius, colors[hand%len(colors)], -1)
            if i % 4 == 0:  # label some points to avoid clutter
                cv2.putText(img, str(i), (px+2, py-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[hand%len(colors)], 1, cv2.LINE_AA)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="video_overlay.mp4")
    parser.add_argument("--max_hands", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)
    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(df))
    print("Frames (video):", total, "CSV rows:", len(df))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(df):
            row = df.iloc[frame_idx]
            frame_annot = draw_landmarks_from_row(frame.copy(), row, max_hands=args.max_hands)
        else:
            frame_annot = frame
        out.write(frame_annot)
        frame_idx += 1

    cap.release()
    out.release()
    print("Overlay saved to:", args.out)

if __name__ == "__main__":
    main()
