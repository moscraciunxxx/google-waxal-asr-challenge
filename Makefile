.PHONY: setup data train-smoke train eval predict submit pipeline-smoke validate-submission test

PYTHON ?= .venv/bin/python
SCRATCH ?= .scratch

setup:
	uv venv .venv --python 3.12
	.venv/bin/uv pip install -r requirements.txt
	.venv/bin/uv pip install pytest

data:
	$(PYTHON) scripts/prepare_data.py

train-smoke:
	$(PYTHON) -m src.train --max-per-lang-split 8 --epochs 1 --output-dir checkpoints/whisper-waxal-smoke

train:
	$(PYTHON) -m src.train --epochs 3 --output-dir checkpoints/whisper-waxal

eval:
	$(PYTHON) -m src.evaluate --checkpoint checkpoints/whisper-waxal/best --out outputs/local_metrics.json

predict:
	$(PYTHON) -m src.infer --checkpoint checkpoints/whisper-waxal/best --split test --out outputs/test_predictions.csv

submit:
	$(PYTHON) -m src.submission --predictions outputs/test_predictions.csv --out submission.csv

pipeline-smoke:
	$(PYTHON) scripts/run_pipeline.py --smoke --scratch "$(SCRATCH)"

pipeline:
	$(PYTHON) scripts/run_pipeline.py --scratch "$(SCRATCH)"

validate-submission:
	$(PYTHON) scripts/validate_public_submission.py

test:
	$(PYTHON) -m pytest tests/ -q
