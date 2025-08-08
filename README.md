

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
