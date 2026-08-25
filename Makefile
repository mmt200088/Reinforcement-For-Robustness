.DEFAULT_GOAL := help
PYTHON ?= python3
PRESET ?= bert-base-mrpc-stage2-rl
ALGORITHM ?= rl

.PHONY: help test lint format docker stage1 stage2 comparator preset-check

help:
	@printf "Available targets: test lint format docker stage1 stage2 comparator preset-check\n"

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	ruff check .

format:
	ruff format .

docker:
	docker build -t rfr:latest .

stage1:
	bash llama_7B_LayerImportance.sh run rl --preset bert-base-mrpc-stage1-rl --fresh

stage2:
	bash llama_7B_LayerImportance.sh run rl --preset $(PRESET) --fresh

comparator:
	bash llama_7B_LayerImportance.sh run $(ALGORITHM) --fresh

preset-check:
	$(PYTHON) tools/validate_preset.py presets/*.conf Paean/presets/*.conf
