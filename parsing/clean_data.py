#!/usr/bin/env python3
import csv
import json
import math
import random
import argparse
import pathlib
import sys
from typing import List, Dict, Tuple, Optional


def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        rows = list(r)
        return r.fieldnames or [], rows


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def detect_numeric_columns(rows: List[Dict[str, str]], fields: List[str], exclude: List[str]) -> List[str]:
    cols = []
    for c in fields:
        if c in exclude:
            continue
        seen = 0
        ok = 0
        for row in rows:
            v = row.get(c, "")
            if v == "" or v is None:
                continue
            seen += 1
            if is_float(v):
                ok += 1
        if seen > 0 and ok / seen > 0.95:
            cols.append(c)
    return cols


def col_stats(values: List[Optional[float]]) -> Tuple[float, float]:
    xs = [x for x in values if x is not None]
    n = len(xs)
    if n == 0:
        return 0.0, 1.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    if std <= 1e-12:
        std = 1.0
    return mean, std


def fit_scaler(rows: List[Dict[str, str]], num_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in num_cols:
        vals: List[Optional[float]] = []
        for row in rows:
            v = row.get(c, "")
            vals.append(float(v) if is_float(v) else None)
        m, s = col_stats(vals)
        stats[c] = {"mean": m, "std": s}
    return stats


def transform_row(row: Dict[str, str], num_cols: List[str], stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in num_cols:
        v = row.get(c, "")
        if not is_float(v):
            x = stats[c]["mean"]  # imputation par moyenne
        else:
            x = float(v)
        out[c] = (x - stats[c]["mean"]) / stats[c]["std"]
    return out


def write_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        # Crée un fichier vide avec juste l'en-tête si possible
        with open(path, "w", newline='', encoding="utf-8") as f:
            pass
        return
    # Conserver l'ordre d'insertion des clés du premier dict
    with open(path, "w", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Nettoyage et normalisation des datasets pour DSLR (solution 1)")
    ap.add_argument("--input", required=True, help="Chemin vers le CSV d'entrée (train ou test)")
    ap.add_argument("--output_dir", default="data", help="Répertoire de sortie des artefacts")
    ap.add_argument("--label_col", default="Hogwarts House", help="Nom de la colonne label (présente uniquement en train)")
    ap.add_argument("--exclude_cols", nargs="*", default=["Index"], help="Colonnes à exclure du traitement numérique")
    ap.add_argument("--passthrough_cols", nargs="*", default=["First Name", "Last Name"], help="Colonnes à recopier telles quelles dans dataset_clean.csv")
    ap.add_argument("--seed", type=int, default=42, help="Graine pour le split train/val")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Part de validation (0-1)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    (out_dir / "split").mkdir(parents=True, exist_ok=True)

    fields, rows = read_csv(args.input)
    if len(rows) == 0:
        print("[clean_data] Aucune donnée trouvée dans l'entrée.")
        return 0

    is_train = args.label_col in fields

    # Détection des colonnes numériques (majoritairement numériques)
    exclude = list(args.exclude_cols)
    if is_train:
        exclude.append(args.label_col)
    # Ne pas considérer les colonnes passthrough comme features numériques
    for c in args.passthrough_cols:
        if c not in exclude:
            exclude.append(c)
    num_cols = detect_numeric_columns(rows, fields, exclude=exclude)

    # Apprentissage des stats pour standardisation
    stats = fit_scaler(rows, num_cols)

    # Transformation des features + ajout des colonnes passthrough (+ label si train)
    X_rows: List[Dict[str, object]] = []
    for r in rows:
        out_row: Dict[str, object] = {}
        # 1) Colonnes d'identification copiées telles quelles (si présentes)
        for c in args.passthrough_cols:
            out_row[c] = r.get(c, "")
        # 2) Features numériques standardisées
        out_row.update(transform_row(r, num_cols, stats))
        # 3) Label (maison) si présent dans le dataset d'entrée
        if is_train:
            out_row[args.label_col] = r.get(args.label_col, "")
        X_rows.append(out_row)

    write_csv(str(out_dir / "dataset_clean.csv"), X_rows)

    # Écriture des labels (si train)
    meta: Dict[str, object] = {
        "numeric_columns": num_cols,
        "scaler": stats,
        "passthrough_columns": args.passthrough_cols,
    }
    if is_train:
        labels = [r.get(args.label_col, "") for r in rows]
        with open(out_dir / "labels.csv", "w", newline='', encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label"])
            w.writerows([[y] for y in labels])
        classes = sorted({y for y in labels if y != ""})
        meta["label_col"] = args.label_col
        meta["classes"] = classes

        # Split train/val déterministe
        idxs = list(range(len(rows)))
        random.Random(args.seed).shuffle(idxs)
        n_val = int(len(idxs) * args.val_ratio)
        val_idx, train_idx = sorted(idxs[:n_val]), sorted(idxs[n_val:])
        (out_dir / "split" / "train_idx.txt").write_text("\n".join(map(str, train_idx)), encoding="utf-8")
        (out_dir / "split" / "val_idx.txt").write_text("\n".join(map(str, val_idx)), encoding="utf-8")

    # Sauvegarde des métadonnées
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[clean_data] Terminé. Artefacts écrits dans: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
