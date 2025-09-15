# inspect_landmarks.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hand(frame_row, max_hands=2):
    # Assumes columns: hand0_x_0, hand0_y_0, hand0_z_0, ..., hand1_x_20...
    coords = []
    for h in range(max_hands):
        x_cols = [c for c in frame_row.index if c.startswith(f"hand{h}_x_")]
        y_cols = [c for c in frame_row.index if c.startswith(f"hand{h}_y_")]
        if not x_cols or not y_cols:
            continue
        xs = frame_row[x_cols].values.astype(float)
        ys = frame_row[y_cols].values.astype(float)
        if np.all(np.isnan(xs)) or np.all(np.isnan(ys)):
            continue
        coords.append((xs, ys, h))
    plt.figure(figsize=(6,6))
    for xs, ys, h in coords:
        plt.scatter(xs, 1-ys, s=30)  # invert y to match image coords
        for i,(x,y) in enumerate(zip(xs,ys)):
            plt.text(x, 1-y, str(i), fontsize=8)
    plt.title("Hand landmarks (normalized image coords)")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    if args.frame < 0 or args.frame >= len(df):
        print("Frame fora do range.")
        return
    row = df.iloc[args.frame]
    plot_hand(row)
    print(row.head(30))
if __name__ == "__main__":
    main()
