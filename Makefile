VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)
DATASETS_DIR := datasets
VISUALIZATION_DIR := src/visualization

.PHONY: prepare-train prepare-test inspect-train inspect-test clean describe describe-train describe-test histogram deps venv train-model train-custom predict-model

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

venv:
	python3 -m venv $(VENV_DIR)
	@echo "Virtualenv créée dans $(VENV_DIR). Activez-la avec: source $(VENV_DIR)/bin/activate"

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

train-model:
	$(PY) src/algorithms/logreg_train.py $(DATASETS_DIR)/dataset_train.csv

train-custom:
	@if [ -z "$(dataset)" ]; then \
		echo "Usage: make train-custom dataset=PATH/TO/DATASET.csv"; \
		echo "   or: make train-custom dataset=PATH/TO/DATASET.csv output=PATH/TO/MODEL.pkl"; \
		exit 1; \
	fi
	$(PY) src/algorithms/logreg_train.py $(dataset) $(if $(output),--output $(output))

predict-model:
	$(PY) src/algorithms/logreg_predict.py

%:
	@:
