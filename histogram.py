#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Tuple, Iterable

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("[histogram][ERR] matplotlib requis. Installez-le avec: python3 -m pip install matplotlib", file=sys.stderr)
    raise

HOUSES_CANON = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
HOUSE_COL = "Hogwarts House"


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Trace des histogrammes par maison pour un cours donné."
    )
    ap.add_argument("csv_path", help="Chemin vers le dataset CSV nettoyé (ex: data/train/dataset_clean.csv)")
    ap.add_argument("-f", "--feature", dest="feature", help="Nom exact de la colonne (ex: Astronomy)")
    ap.add_argument("--bins", type=int, default=30, help="Nombre de bins (par défaut: 30)")
    ap.add_argument("--no-show", action="store_true", help="Ne pas afficher la fenêtre Matplotlib")
    ap.add_argument("--out-dir", default="data", help="Dossier de sortie des images (par défaut: data)")
    return ap.parse_args(argv[1:])


def file_is_readable(path: str) -> bool:
    try:
        st = os.stat(path)
        return os.path.isfile(path) and st.st_size >= 0
    except OSError:
        return False


def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fields = rdr.fieldnames or []
        return fields, rows


def safe_float(s: str) -> float | None:
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.lower() in {"nan", "na", "none"}:
        return None
    try:
        x = float(t)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def detect_numeric_columns(cols: List[str], rows: List[Dict[str, str]]) -> List[str]:
    num_cols: List[str] = []
    for c in cols:
        if c == HOUSE_COL:
            continue
        total_non_empty = 0
        total_numeric = 0
        for r in rows:
            v = r.get(c, "")
            if v is None or str(v).strip() == "":
                continue
            total_non_empty += 1
            if safe_float(v) is not None:
                total_numeric += 1
        if total_non_empty > 0 and total_numeric == total_non_empty:
            num_cols.append(c)
    return num_cols


def values_by_house(rows: List[Dict[str, str]], feature: str) -> Dict[str, List[float]]:
    by_h: Dict[str, List[float]] = {h: [] for h in HOUSES_CANON}
    for r in rows:
        h = (r.get(HOUSE_COL) or "").strip()
        x = safe_float(r.get(feature, ""))
        if h in by_h and x is not None:
            by_h[h].append(x)
    return by_h


def choose_most_homogeneous_feature(rows: List[Dict[str, str]], numeric_cols: List[str]) -> str | None:
    # Heuristique simple: minimiser l'écart-type des moyennes par maison, normalisé par l'écart-type global
    best_feature = None
    best_score = float("inf")
    eps = 1e-9

    for c in numeric_cols:
        by_h = values_by_house(rows, c)
        all_vals: List[float] = []
        means: List[float] = []
        for h in HOUSES_CANON:
            xs = by_h[h]
            all_vals.extend(xs)
            if len(xs) == 0:
                # pénalise les features trop vides
                means.append(float("nan"))
            else:
                s = 0.0
                for v in xs:
                    s += v
                means.append(s / len(xs))
        # filtre si insuffisant
        if len(all_vals) < 4 or any(isinstance(m, float) and not math.isfinite(m) for m in means):
            continue
        # std global
        mu = sum(all_vals) / len(all_vals)
        var = 0.0
        for v in all_vals:
            d = v - mu
            var += d * d
        var /= len(all_vals)
        std_global = math.sqrt(var)
        if std_global <= eps:
            score = 0.0
        else:
            # std des moyennes par maison
            mu_m = sum(means) / len(means)
            var_m = 0.0
            for m in means:
                d = m - mu_m
                var_m += d * d
            var_m /= len(means)
            std_means = math.sqrt(var_m)
            score = std_means / (std_global + eps)
        if score < best_score:
            best_score = score
            best_feature = c
    return best_feature


def common_bin_edges(values: Iterable[float], bins: int) -> List[float]:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return []
    vmin = min(vals)
    vmax = max(vals)
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return []
    if vmax == vmin:
        # élargit un peu pour éviter un seul point
        vmin -= 0.5
        vmax += 0.5
    width = (vmax - vmin) / float(max(1, int(bins)))
    edges = [vmin + i * width for i in range(int(bins) + 1)]
    return edges


def slugify(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def plot_histograms(by_house: Dict[str, List[float]], feature: str, bins: int, out_dir: str, show: bool) -> str | None:
    all_vals: List[float] = []
    for h in HOUSES_CANON:
        all_vals.extend(by_house.get(h, []))
    edges = common_bin_edges(all_vals, bins)
    if not edges:
        print(f"[histogram][ERR] Impossible de déterminer les bins pour '{feature}'.", file=sys.stderr)
        return None

    colors = {
        "Gryffindor": "#d62728",
        "Slytherin": "#2ca02c",
        "Ravenclaw": "#1f77b4",
        "Hufflepuff": "#ffbf00",
    }

    plt.figure(figsize=(9, 6))
    for h in HOUSES_CANON:
        xs = by_house.get(h, [])
        if not xs:
            continue
        plt.hist(xs, bins=edges, density=True, alpha=0.5, label=h, color=colors.get(h, None))

    plt.title(f"Distribution des scores par maison — {feature}")
    plt.xlabel(feature)
    plt.ylabel("Densité")
    plt.legend()
    plt.tight_layout()

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"[histogram][ERR] Impossible de créer le dossier '{out_dir}': {e}", file=sys.stderr)
        return None

    out_path = os.path.join(out_dir, f"histogram_{slugify(feature)}.png")
    try:
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        print(f"[histogram][ERR] Échec de sauvegarde de '{out_path}': {e}", file=sys.stderr)
        return None

    if not show:
        plt.close()
    else:
        try:
            plt.show()
        except Exception:
            # En environnement headless, on a quand même le PNG
            plt.close()
    return out_path


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    csv_path = args.csv_path

    if not file_is_readable(csv_path):
        print(f"[histogram][ERR] Fichier introuvable: {csv_path}", file=sys.stderr)
        return 2

    cols, rows = read_csv(csv_path)
    if not cols:
        print("[histogram][ERR] CSV invalide (en-tête manquante).", file=sys.stderr)
        return 2

    if HOUSE_COL not in cols:
        print(f"[histogram][ERR] Colonne '{HOUSE_COL}' absente. Assurez-vous d'utiliser le CSV nettoyé incluant le label.", file=sys.stderr)
        return 2

    numeric_cols = detect_numeric_columns(cols, rows)
    if not numeric_cols:
        print("[histogram] Aucune colonne numérique détectée.", file=sys.stderr)
        return 2

    feature = args.feature
    if feature:
        if feature not in numeric_cols:
            print(f"[histogram][ERR] Feature inconnue ou non numérique: '{feature}'.", file=sys.stderr)
            print("Features numériques disponibles:", ", ".join(numeric_cols), file=sys.stderr)
            return 2
    else:
        feature = choose_most_homogeneous_feature(rows, numeric_cols)
        if not feature:
            print("[histogram][ERR] Impossible de déterminer automatiquement une feature homogène.", file=sys.stderr)
            return 2
        print(f"[histogram] Feature auto-sélectionnée: {feature}")

    by_h = values_by_house(rows, feature)
    out_path = plot_histograms(by_h, feature, args.bins, args.out_dir, show=(not args.no_show))
    if out_path:
        print(f"[histogram] Image sauvegardée: {out_path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
