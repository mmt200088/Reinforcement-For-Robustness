.DEFAULT_GOAL := help
PYTHON ?= python3
PRESET ?= bert-base-mrpc-stage2-rl
ALGORITHM ?= rl

.PHONY: help lint format docker stage1 stage2 comparator preset-check

help:
	@printf "Available targets: lint format docker stage1 stage2 comparator preset-check\n"

lint:
	ruff check .

format:
	ruff format .

docker:
	docker build -t rfr:latest .

stage1:
	bash run_search.sh run rl --preset bert-base-mrpc-stage1-rl --fresh

stage2:
	bash run_search.sh run rl --preset $(PRESET) --fresh

comparator:
	bash run_search.sh run $(ALGORITHM) --fresh

preset-check:
	PYTHONPATH=src $(PYTHON) -m rfr.cli.validate_preset \
		configs/presets/*.conf configs/evaluation/presets/*.conf
