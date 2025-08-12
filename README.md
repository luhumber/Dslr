## Tester rapidement le parsing et inspecter les sorties

Commandes utiles:

```bash
# Préparer les artefacts train/test
make prepare-train
make prepare-test

# Inspecter les artefacts (aperçu des lignes, stats, distribution des labels)
make inspect-train
make inspect-test
```

Sans Makefile:

```bash
python3 parsing/clean_data.py --input datasets/dataset_train.csv --output_dir data/train
python3 parsing/inspect_artifacts.py --dir data/train --rows 5 --stats
```

Nettoyage:

```bash
make clean
```

---

## Décrire un dataset (describe.py)

Le script `describe.py` calcule manuellement (sans pandas/numpy) les statistiques descriptives pour toutes les colonnes numériques d’un CSV: Count, Mean, Std, Min, 25%, 50%, 75%, Max.

Affichage dans le terminal et export automatique d’un CSV des résultats sous `data/describe_<nom_du_fichier>.csv`.

Exemples d’utilisation:

```bash
# Avec variable positionnelle (conseillé)
make describe datasets/dataset_train.csv
make describe datasets/dataset_test.csv

# Ou avec une variable nommée
make describe file=datasets/dataset_test.csv

# Appel direct sans make
python3 describe.py datasets/dataset_test.csv
```

Notes:
- Les colonnes non numériques ou avec valeurs non finies (NaN, inf) sont ignorées.
- L’écart-type est calculé en mode population (ddof=0).
- Le fichier de sortie est créé sous `data/` (créé automatiquement si absent).
