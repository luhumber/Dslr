PY := python3
DATASETS_DIR := datasets

.PHONY: prepare-train prepare-test inspect-train inspect-test clean

prepare-train:
	$(PY) parsing/clean_data.py --input $(DATASETS_DIR)/dataset_train.csv --output_dir data/train

prepare-test:
	$(PY) parsing/clean_data.py --input $(DATASETS_DIR)/dataset_test.csv --output_dir data/test

inspect-train:
	$(PY) parsing/inspect_artifacts.py --dir data/train --rows 5 --stats

inspect-test:
	$(PY) parsing/inspect_artifacts.py --dir data/test --rows 5 --stats

clean:
	rm -rf data
