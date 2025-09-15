#!/usr/bin/env python3
"""
inspect_landmarks_debug.py

Script de diagnóstico robusto para verificar por que o inspect não está mostrando nada.
- Imprime info do ambiente (Python, cwd)
- Lista arquivos na pasta atual
- Verifica existência e tamanho do CSV
- Mostra primeiras 1kB do CSV (raw)
- Tenta ler com pandas e imprime shape/colunas
- Mostra um resumo do frame pedido
- Tenta salvar imagem do frame (se landmarks presentes)
- Salva tudo em inspect_log.txt

Uso:
python inspect_landmarks_debug.py --csv "caminho/para/landmarks.csv" --frame 0 --out frame_debug.png
"""
import argparse, os, sys, traceback, io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # evita janelas gráficas que podem não abrir
import matplotlib.pyplot as plt

LOGFILE = "inspect_log.txt"

def write_log(s, to_console=True):
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")
    if to_console:
        print(s)

def safe_head_raw(path, nbytes=1024):
    try:
        with open(path, "rb") as f:
            raw = f.read(nbytes)
        return raw
    except Exception as e:
        return f"ERRO ao ler raw: {e}"

def plot_hand_and_save(row, out_path, max_hands=2):
    coords = []
    for h in range(max_hands):
        x_cols = [c for c in row.index if c.startswith(f"hand{h}_x_")]
        y_cols = [c for c in row.index if c.startswith(f"hand{h}_y_")]
        if not x_cols or not y_cols:
            continue
        try:
            xs = row[x_cols].astype(float).values
            ys = row[y_cols].astype(float).values
        except Exception as e:
            write_log(f"Erro convertendo coords mão {h}: {e}")
            continue
        if np.all(np.isnan(xs)) or np.all(np.isnan(ys)):
            continue
        coords.append((xs, ys, h))
    if not coords:
        write_log("Nenhuma mão detectada neste frame (ou colunas ausentes).")
        return False
    plt.figure(figsize=(6,6))
    for xs, ys, h in coords:
        plt.scatter(xs, 1-ys, s=30)
        for i,(x,y) in enumerate(zip(xs,ys)):
            plt.text(x, 1-y, str(i), fontsize=8)
    plt.title("Hand landmarks (normalized coords)")
    plt.axis('off')
    try:
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        write_log(f"Imagem salva em: {out_path}")
        return True
    except Exception as e:
        write_log(f"Erro salvando imagem: {e}")
        return False

def main():
    # limpa log antigo
    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="frame_debug.png")
    args = parser.parse_args()

    write_log("=== INSPECT DEBUG START ===")
    write_log(f"Python executable: {sys.executable}")
    write_log(f"Python version: {sys.version.replace(chr(10),' ')}")
    write_log(f"Current working dir: {os.getcwd()}")
    write_log("Files in cwd:")
    for f in os.listdir("."):
        write_log(f"  - {f}")

    csv_path = args.csv
    write_log(f"CSV path requested: {csv_path}")
    write_log(f"CSV exists? {os.path.exists(csv_path)}")
    if os.path.exists(csv_path):
        try:
            write_log(f"CSV absolute path: {os.path.abspath(csv_path)}")
            size = os.path.getsize(csv_path)
            write_log(f"CSV size (bytes): {size}")
            raw = safe_head_raw(csv_path, nbytes=2048)
            if isinstance(raw, bytes):
                try:
                    raw_text = raw.decode("utf-8", errors="replace")
                except:
                    raw_text = str(raw)
            else:
                raw_text = str(raw)
            write_log("=== First ~2KB of CSV (raw) ===")
            for line in raw_text.splitlines()[:40]:
                write_log(line)
            write_log("=== End raw preview ===")
        except Exception as e:
            write_log(f"Erro ao inspecionar CSV: {e}")
    else:
        write_log("CSV não existe no caminho informado. Abortando após logs.")
        write_log("=== INSPECT DEBUG END ===")
        return

    # tenta ler com pandas
    try:
        df = pd.read_csv(csv_path)
        write_log(f"pandas leu CSV com shape: {df.shape}")
        write_log(f"Colunas (primeiras 80): {list(df.columns[:80])}")
    except Exception as e:
        write_log("ERRO lendo CSV com pandas:")
        write_log(traceback.format_exc())
        write_log("Tentando leitura manual linha a linha...")
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [next(f) for _ in range(10)]
            write_log("Primeiras 10 linhas (manual):")
            for L in lines:
                write_log(L.rstrip("\n"))
        except Exception as e2:
            write_log(f"Falha leitura manual: {e2}")
        write_log("=== INSPECT DEBUG END ===")
        return

    # quick checks
    if 'frame' in df.columns:
        write_log("Coluna 'frame' encontrada.")
    pres_cols = [c for c in df.columns if c.endswith("_presence")]
    write_log(f"Presença de '*_presence' cols: {pres_cols}")

    if args.frame < 0 or args.frame >= len(df):
        write_log(f"Frame pedido ({args.frame}) fora do range [0, {len(df)-1}]")
        write_log("=== INSPECT DEBUG END ===")
        return

    row = df.iloc[args.frame]
    write_log(f"Mostrando primeiras 60 colunas do frame {args.frame}:")
    for c in df.columns[:60]:
        write_log(f"  {c}: {row.get(c)}")

    # salva imagem do frame (se possível)
    ok = plot_hand_and_save(row, args.out)
    if not ok:
        write_log("Não foi possível gerar imagem com landmarks (ou eles estão ausentes).")

    write_log("=== INSPECT DEBUG END ===")

if __name__ == "__main__":
    main()
