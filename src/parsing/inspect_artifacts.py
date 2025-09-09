import argparse
import csv
import json
import pathlib
from typing import Dict, List, Tuple, Optional
import math


def read_csv_rows(path: pathlib.Path, limit: Optional[int] = None) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for i, row in enumerate(r):
            rows.append(row)
            if limit is not None and i + 1 >= limit:
                break
        return r.fieldnames or [], rows


def read_all_csv_rows(path: pathlib.Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        rows = list(r)
        return r.fieldnames or [], rows


def compute_stats_numeric(cols: List[str], rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    sums: Dict[str, float] = {c: 0.0 for c in cols}
    counts: Dict[str, int] = {c: 0 for c in cols}
    mins: Dict[str, float] = {c: float('inf') for c in cols}
    maxs: Dict[str, float] = {c: float('-inf') for c in cols}
    for row in rows:
        for c in cols:
            v = row.get(c, "")
            try:
                x = float(v)
            except Exception:
                continue
            sums[c] += x
            counts[c] += 1
            if x < mins[c]:
                mins[c] = x
            if x > maxs[c]:
                maxs[c] = x
    means: Dict[str, float] = {}
    for c in cols:
        n = counts[c] if counts[c] > 0 else 1
        means[c] = sums[c] / n
    var_sums: Dict[str, float] = {c: 0.0 for c in cols}
    for row in rows:
        for c in cols:
            v = row.get(c, "")
            try:
                x = float(v)
            except Exception:
                continue
            var_sums[c] += (x - means[c]) ** 2
    stats: Dict[str, Dict[str, float]] = {}
    for c in cols:
        n = counts[c] if counts[c] > 0 else 1
        var = var_sums[c] / n
        std = math.sqrt(var)
        mn = mins[c] if mins[c] != float('inf') else float('nan')
        mx = maxs[c] if maxs[c] != float('-inf') else float('nan')
        stats[c] = {"count": float(counts[c]), "mean": means[c], "std": std, "min": mn, "max": mx}
    return stats


def preview_rows(cols: List[str], rows: List[Dict[str, str]], n: int) -> str:
    out = []
    header = ", ".join(cols)
    out.append(header)
    for i, row in enumerate(rows[:n]):
        out.append(", ".join(str(row.get(c, "")) for c in cols))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Inspection rapide des artefacts générés par clean_data.py")
    ap.add_argument("--dir", required=True, help="Répertoire d'artefacts (output_dir de clean_data.py)")
    ap.add_argument("--rows", type=int, default=5, help="Nombre de lignes à prévisualiser")
    ap.add_argument("--stats", action="store_true", help="Recalculer des stats (mean/std/min/max) sur dataset_clean.csv")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    x_path = root / "dataset_clean.csv"
    y_path = root / "labels.csv"
    meta_path = root / "metadata.json"
    train_idx_path = root / "split" / "train_idx.txt"
    val_idx_path = root / "split" / "val_idx.txt"

    if not x_path.exists():
        print(f"[inspect] Fichier introuvable: {x_path}")
        return 2

    meta: Dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"[inspect] Impossible de lire metadata.json: {e}")

    cols, head_rows = read_csv_rows(x_path, limit=args.rows)
    all_cols, all_rows = read_all_csv_rows(x_path)

    print("=== Aperçu dataset_clean.csv ===")
    print(f"Chemin: {x_path}")
    print(f"Lignes: {len(all_rows)} | Colonnes: {len(all_cols)}")
    if meta.get("numeric_columns"):
        print(f"Colonnes numériques (metadata): {len(meta['numeric_columns'])}")
    if all_cols:
        print(preview_rows(all_cols, head_rows, args.rows))

    if args.stats and all_cols:
        print("\n=== Stats rapides (recalculées) ===")
        stats = compute_stats_numeric(all_cols, all_rows)
        show_cols = all_cols[: min(8, len(all_cols))]
        for c in show_cols:
            s = stats[c]
            print(f"{c}: count={int(s['count'])}, mean={s['mean']:.3f}, std={s['std']:.3f}, min={s['min']:.3f}, max={s['max']:.3f}")
        if len(all_cols) > len(show_cols):
            print(f"... ({len(all_cols) - len(show_cols)} colonnes supplémentaires)")

    if y_path.exists():
        _, y_rows = read_all_csv_rows(y_path)
        print("\n=== Labels (labels.csv) ===")
        print(f"Lignes labels: {len(y_rows)}")
        dist: Dict[str, int] = {}
        for r in y_rows:
            y = r.get("label", "")
            dist[y] = dist.get(y, 0) + 1
        total = sum(dist.values()) or 1
        for k in sorted(dist.keys()):
            print(f"- {k}: {dist[k]} ({100.0*dist[k]/total:.1f}%)")
        if meta.get("classes"):
            print(f"Classes (metadata): {', '.join(meta['classes'])}")
        if len(y_rows) != len(all_rows):
            print(f"[inspect][WARN] Nombre de labels ({len(y_rows)}) != nombre de lignes X ({len(all_rows)})")

    if train_idx_path.exists() and val_idx_path.exists():
        train_idx = [int(x) for x in train_idx_path.read_text(encoding='utf-8').strip().splitlines() if x.strip()]
        val_idx = [int(x) for x in val_idx_path.read_text(encoding='utf-8').strip().splitlines() if x.strip()]
        print("\n=== Split ===")
        print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
        inter = set(train_idx).intersection(val_idx)
        if inter:
            print(f"[inspect][WARN] Indices communs entre train et val: {len(inter)}")
        maxi = max(train_idx + val_idx) if (train_idx or val_idx) else -1
        if maxi >= len(all_rows):
            print(f"[inspect][WARN] Index max {maxi} >= nb lignes {len(all_rows)}")

    if meta:
        print("\n=== Metadata ===")
        for k in ["label_col", "classes"]:
            if k in meta:
                print(f"{k}: {meta[k]}")
        if "scaler" in meta and isinstance(meta["scaler"], dict):
            print(f"scaler: {len(meta['scaler'])} colonnes")

    print("\n[inspect] Terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
