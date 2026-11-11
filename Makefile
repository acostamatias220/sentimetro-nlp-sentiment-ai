# Sentímetro · ML pipeline
#   make install      -> virtualenv with train + dev dependencies (uv)
#   make data         -> download (pinned revision) + clean/dedupe/split
#   make train        -> TF-IDF baseline + fine-tuned DistilBETO (tracked in MLflow)
#   make export       -> ONNX fp32 + int8 with parity checks -> models/production
#   make evaluate     -> test metrics, confusion matrices, error analysis, CPU benchmark -> reports/
#   make serve        -> FastAPI on http://127.0.0.1:8000 (demo at /demo/, docs at /docs)
# Extra CLI overrides: make train-transformer ARGS="--set transformer.epochs=3"

UV      ?= uv
RUN     := $(UV) run --frozen
CLI     := $(RUN) sentimetro $(ARGS)
PORT    ?= 8000

RAW       := data/raw/train.parquet
PROCESSED := data/processed/train.parquet
BASELINE  := models/baseline/model_card.json
TRANSFORMER := models/transformer/final/model_card.json
PRODUCTION  := models/production/model_card.json

.PHONY: help install data train train-baseline train-baseline-30k train-transformer train-transformer-30k export \
        evaluate pipeline serve mlflow-ui test test-fast lint format typecheck check \
        docker-build docker-up clean clean-models

help: ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## create .venv with all extras and dev tools, install git hooks
	$(UV) sync --frozen --all-extras
	$(RUN) pre-commit install

$(RAW):
	$(CLI) download

$(PROCESSED): $(RAW) src/sentimetro/data/preprocess.py src/sentimetro/text.py
	$(CLI) preprocess

data: $(PROCESSED) ## download + preprocess the dataset

train-baseline: $(PROCESSED) ## TF-IDF + logistic regression on the full train split
	$(CLI) train-baseline

train-baseline-30k: $(PROCESSED) ## same baseline on the 30k subsample used by the transformer
	$(CLI) --set baseline.train_samples=30000 --set baseline.output_dir=models/baseline-30k train-baseline

train-transformer: $(PROCESSED) ## fine-tune DistilBETO (CPU defaults in configs/params.yaml)
	$(CLI) train-transformer

train-transformer-30k: $(PROCESSED) ## ablation: 30k reviews, 2 epochs, unweighted loss
	$(CLI) --set transformer.train_samples=30000 --set transformer.epochs=2 \
		--set transformer.eval_steps=400 --set transformer.class_weighting=null \
		--set transformer.output_dir=models/transformer-30k train-transformer

train: train-baseline train-baseline-30k train-transformer ## train the models used in production and the baselines

export: ## export the transformer to ONNX (fp32 + int8) and publish models/production
	$(CLI) export

evaluate: ## evaluate every trained model on the test split -> reports/
	$(CLI) evaluate

pipeline: data train export evaluate ## run everything end to end

serve: ## run the API (uses models/production)
	$(RUN) sentimetro serve --port $(PORT)

mlflow-ui: ## browse experiments at http://127.0.0.1:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test: ## all tests (includes the tiny ONNX export tests, needs the train extra)
	$(RUN) pytest --cov --cov-report=term-missing

test-fast: ## tests without torch
	$(RUN) pytest -m "not slow"

lint: ## ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## auto-format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck: ## mypy (strict)
	$(RUN) mypy

check: lint typecheck test ## everything CI runs

docker-build: ## build the inference image
	docker build -t sentimetro-api .

docker-up: ## API + Prometheus (needs models/production)
	docker compose up --build

clean: ## remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

clean-models: ## remove trained models and MLflow runs (data is kept)
	rm -rf models mlflow.db mlartifacts
