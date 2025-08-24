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

---

## Histogramme par maison (histogram.py)

Le script `histogram.py` trace un histogramme des notes par maison pour un cours donné et aide à répondre à:
“Which Hogwarts course has a homogeneous score distribution between all four houses?”

Fonctionnalités:
- Détection automatique des colonnes de cours numériques.
- Choix automatique du cours le plus homogène si `--feature` n’est pas fourni.
- Histogrammes superposés (bins communs, densité) pour les 4 maisons.
- Sauvegarde en PNG sous `data/histogram_<feature>.png`.

Installation de la dépendance:

```bash
python3 -m pip install matplotlib
```

Exemples d’utilisation:

```bash
python3 histogram.py datasets/dataset_train.csv              # auto-choisit le cours
python3 histogram.py datasets/dataset_train.csv -f Astronomy  # force une feature
python3 histogram.py datasets/dataset_train.csv -f Astronomy --bins 40 --no-show
```

Astuce:
- Les cours possibles sont détectés en ignorant les colonnes non numériques ou métadonnées (Index, nom, etc.).
