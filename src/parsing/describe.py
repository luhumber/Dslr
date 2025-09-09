from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Tuple

ROW_LABELS = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Affiche des stats descriptives pour toutes les features numériques d'un CSV (sans libs)."
    )
    ap.add_argument("csv_path", help="Chemin vers le dataset CSV (ex: datasets/dataset_train.csv)")
    return ap.parse_args(argv[1:])


def file_is_readable(path: str) -> bool:
    try:
        st = os.stat(path)
        return os.path.isfile(path) and st.st_size >= 0
    except OSError:
        return False


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


def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        try:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        except csv.Error as e:
            print(f"[describe][ERR] CSV invalide: {e}", file=sys.stderr)
            return [], []
        fields = rdr.fieldnames or []
        return fields, rows


def detect_numeric_columns(cols: List[str], rows: List[Dict[str, str]]) -> List[str]:
    num_cols: List[str] = []
    for c in cols:
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


def percentile_linear(sorted_xs: List[float], p: float) -> float:
    n = len(sorted_xs)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_xs[0]
    k = (n - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_xs[f]
    return sorted_xs[f] * (c - k) + sorted_xs[c] * (k - f)


def describe_for_cols(rows: List[Dict[str, str]], cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in cols:
        xs: List[float] = []
        for r in rows:
            x = safe_float(r.get(c, ""))
            if x is not None:
                xs.append(x)
        n = len(xs)
        if n == 0:
            stats[c] = {
                "count": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p25": float("nan"),
                "p50": float("nan"),
                "p75": float("nan"),
                "max": float("nan"),
            }
            continue

        s = 0.0
        i = 0
        while i < n:
            s += xs[i]
            i += 1
        mean = s / n

        var_sum = 0.0
        i = 0
        while i < n:
            d = xs[i] - mean
            var_sum += d * d
            i += 1
        var = var_sum / n
        std = math.sqrt(var)

        mn = xs[0]
        mx = xs[0]
        i = 1
        while i < n:
            v = xs[i]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
            i += 1

        xs_sorted = sorted(xs)
        p25 = percentile_linear(xs_sorted, 0.25)
        p50 = percentile_linear(xs_sorted, 0.50)
        p75 = percentile_linear(xs_sorted, 0.75)

        stats[c] = {
            "count": float(n),
            "mean": mean,
            "std": std,
            "min": mn,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "max": mx,
        }
    return stats


def format_table(cols: List[str], stats: Dict[str, Dict[str, float]]) -> str:
    def fmt(x: float) -> str:
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return f"{x:.6f}"

    key_map = {
        "Count": "count",
        "Mean": "mean",
        "Std": "std",
        "Min": "min",
        "25%": "p25",
        "50%": "p50",
        "75%": "p75",
        "Max": "max",
    }

    vals_by_row = {label: [fmt(stats[c][key_map[label]]) for c in cols] for label in ROW_LABELS}

    col_widths: List[int] = []
    for i, c in enumerate(cols):
        max_val_len = 0
        for label in ROW_LABELS:
            vlen = len(vals_by_row[label][i])
            if vlen > max_val_len:
                max_val_len = vlen
        width = max(len(c), max_val_len)
        col_widths.append(width)

    row_label_width = 0
    for l in ROW_LABELS:
        if len(l) > row_label_width:
            row_label_width = len(l)

    lines: List[str] = []
    header = (" " * (row_label_width + 1)) + " ".join(
        c.ljust(col_widths[i]) for i, c in enumerate(cols)
    )
    lines.append(header)

    for label in ROW_LABELS:
        line = label.ljust(row_label_width) + " " + " ".join(
            vals_by_row[label][i].ljust(col_widths[i]) for i in range(len(cols))
        )
        lines.append(line)
    return "\n".join(lines)


def write_csv_result(input_csv: str, cols: List[str], stats: Dict[str, Dict[str, float]], out_dir: str = "data") -> str | None:
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"[describe][ERR] Impossible de créer le dossier de sortie '{out_dir}': {e}", file=sys.stderr)
        return None

    base = os.path.splitext(os.path.basename(input_csv))[0]
    out_path = os.path.join(out_dir, f"describe_{base}.csv")

    key_map = {
        "Count": "count",
        "Mean": "mean",
        "Std": "std",
        "Min": "min",
        "25%": "p25",
        "50%": "p50",
        "75%": "p75",
        "Max": "max",
    }

    def fmt(x: float) -> str:
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return f"{x:.6f}"

    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stat", *cols])
            for label in ROW_LABELS:
                row = [label]
                for c in cols:
                    row.append(fmt(stats[c][key_map[label]]))
                w.writerow(row)
        return out_path
    except OSError as e:
        print(f"[describe][ERR] Échec d'écriture CSV '{out_path}': {e}", file=sys.stderr)
        return None


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    path = args.csv_path

    if not file_is_readable(path):
        print(f"[describe][ERR] Fichier introuvable ou illisible: {path}", file=sys.stderr)
        return 2

    cols, rows = read_csv(path)
    if not cols:
        print("[describe][ERR] En-tête CSV absente ou illisible.", file=sys.stderr)
        return 2
    if not rows:
        print("[describe][WARN] CSV vide ou sans lignes de données.")
        return 0

    num_cols = detect_numeric_columns(cols, rows)
    if not num_cols:
        print("[describe] Aucune colonne numérique détectée.")
        return 0

    stats = describe_for_cols(rows, num_cols)
    print(format_table(num_cols, stats))

    out_path = write_csv_result(path, num_cols, stats, out_dir="data")
    if out_path:
        print(f"[describe] Résultat CSV écrit: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
