# Virtualenv-aware Python
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)
DATASETS_DIR := datasets
VISUALIZATION_DIR := src/visualization

.PHONY: prepare-train prepare-test inspect-train inspect-test clean describe describe-train describe-test histogram deps venv

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
	$(PY) parsing/describe.py "$$1"

describe-train:
	$(PY) parsing/describe.py $(DATASETS_DIR)/dataset_train.csv

describe-test:
	$(PY) parsing/describe.py $(DATASETS_DIR)/dataset_test.csv

histogram:
	@CSV_PATH="$(CSV)"; \
	if [ -z "$$CSV_PATH" ] && [ -n "$(file)" ]; then CSV_PATH="$(file)"; fi; \
	if [ -z "$$CSV_PATH" ]; then CSV_PATH="$(word 2,$(MAKECMDGOALS))"; fi; \
	if [ -z "$$CSV_PATH" ]; then \
		echo "Usage: make histogram <csv> [OPTS='-f Feature --bins 30 [--no-show]']  OR  make histogram file=PATH/TO.csv"; \
		echo "Exemples:"; \
		echo "  make histogram data/train/dataset_clean.csv OPTS=\"-f Astronomy --bins 40 --no-show\""; \
		echo "  make histogram file=data/train/dataset_clean.csv OPTS=\"--no-show\""; \
		exit 2; \
	fi; \
	echo "Running: $(PY) histogram.py '$$CSV_PATH' $(OPTS)"; \
	$(PY) histogram.py "$$CSV_PATH" $(OPTS)

# Créer une virtualenv locale
venv:
	python3 -m venv $(VENV_DIR)
	@echo "Virtualenv créée dans $(VENV_DIR). Activez-la avec: source $(VENV_DIR)/bin/activate"

# Installer les dépendances Python dans la venv
deps:
	@if [ ! -x "$(VENV_PY)" ]; then \
		echo "[deps] Aucune venv détectée, création dans $(VENV_DIR)..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

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
