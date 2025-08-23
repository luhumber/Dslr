PY := python3
DATASETS_DIR := datasets
VISUALIZATION_DIR := src/visualization

.PHONY: prepare-train prepare-test inspect-train inspect-test clean describe describe-train describe-test

# Permet de passer un argument positionnel après la cible, ex:
#   make describe datasets/dataset_test.csv
# Sans que make essaie de construire ce chemin comme une cible.
# (On filtre explicitement le nom de la cible « describe »)
ARGS = $(filter-out describe,$(MAKECMDGOALS))

prepare-train:
	$(PY) src/parsing/clean_data.py --input $(DATASETS_DIR)/dataset_train.csv --output_dir data/train

prepare-test:
	$(PY) src/parsing/clean_data.py --input $(DATASETS_DIR)/dataset_test.csv --output_dir data/test

inspect-train:
	$(PY) src/parsing/inspect_artifacts.py --dir data/train --rows 5 --stats

inspect-test:
	$(PY) src/parsing/inspect_artifacts.py --dir data/test --rows 5 --stats

clean:
	rm -rf data output

describe:
	@set -- $(ARGS); \
	if [ $$# -eq 0 ] && [ -n "$(file)" ]; then set -- "$(file)"; fi; \
	if [ $$# -eq 0 ]; then \
		echo "Usage: make describe <csv>  OR  make describe file=PATH/TO.csv"; \
		exit 2; \
	fi; \
	$(PY) describe.py "$$1"

describe-train:
	$(PY) describe.py $(DATASETS_DIR)/dataset_train.csv

describe-test:
	$(PY) describe.py $(DATASETS_DIR)/dataset_test.csv

visualizer:
	$(PY) $(VISUALIZATION_DIR)/visualization.py all

visualizer-hist:
	$(PY) $(VISUALIZATION_DIR)/visualization.py hist

visualizer-scatter:
	$(PY) $(VISUALIZATION_DIR)/visualization.py scatter

visualizer-pair:
	$(PY) $(VISUALIZATION_DIR)/visualization.py pair

# Règle générique pour empêcher make d'interpréter l'argument comme une cible à construire
%:
	@:
