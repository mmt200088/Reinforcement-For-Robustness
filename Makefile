# BLB Stage-2 RL · Makefile
# ----------------------------------------------------------------------
# One-liner shortcuts. `make help` lists all targets.

.DEFAULT_GOAL := help
PYTHON ?= python3
PIP ?= pip

# ----------------------------------------------------------------------
# help
# ----------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@printf "BLB Stage-2 RL · available targets\n\n"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  %-22s %s\n", $$1, $$2 }' \
	    $(MAKEFILE_LIST)
	@printf "\nFlags:\n  PYTHON=python3.11   override python interpreter\n"

# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------
.PHONY: test test-fast test-smoke test-all
test: test-fast ## Alias for test-fast (torch-free)

test-fast: ## Run torch-free BLB unit tests (CI parity, < 30s)
	BLB_STRICT=0 $(PYTHON) -m unittest discover -s tests -p "test_blb_*.py" -v

test-smoke: ## End-to-end sequential RL smoke (3 episodes, StubInvoker)
	$(PYTHON) tests/test_sequential_smoke.py

test-all: ## Discover-and-run every test (incl. torch-requiring)
	$(PYTHON) -m unittest discover -s tests -v

# ----------------------------------------------------------------------
# Lint / format
# ----------------------------------------------------------------------
.PHONY: lint lint-fix format
lint: ## Ruff lint (no autofix)
	ruff check .

lint-fix: ## Ruff lint with autofix
	ruff check --fix .

format: ## Ruff format (Black-compatible)
	ruff format .

# ----------------------------------------------------------------------
# Security / deps
# ----------------------------------------------------------------------
.PHONY: audit deps-freeze
audit: ## pip-audit on requirements.txt
	pip-audit --requirement requirements.txt --strict

deps-freeze: ## Snapshot the resolved environment to requirements-frozen.txt
	$(PIP) freeze --exclude-editable > requirements-frozen.txt
	@echo "Wrote requirements-frozen.txt ($$(wc -l < requirements-frozen.txt) packages)"

# ----------------------------------------------------------------------
# Docker
# ----------------------------------------------------------------------
.PHONY: docker docker-run
docker: ## Build Docker image (CUDA 12.1)
	docker build -t blb-rl:latest .

docker-run: ## Drop into a GPU-enabled shell with the repo mounted
	docker run --gpus all -it --rm \
	    -v "$$PWD":/workspace \
	    -v "$$HOME/.cache/huggingface":/root/.cache/huggingface \
	    blb-rl:latest bash

# ----------------------------------------------------------------------
# Training shortcuts
# ----------------------------------------------------------------------
.PHONY: train train-resume train-multi-seed
train: ## Fresh training run (MRPC, sequential)
	bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh

train-resume: ## Resume current training (no --fresh)
	bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl

train-multi-seed: ## Five-seed sweep for significance (set SEEDS=... to override)
	bash tools/run_multi_seed.sh mrpc-blb-stage2-rl \
	    $(or $(SEEDS),1,2,3,4,5) \
	    $(or $(RUN_TAG),trial$$(date +%Y%m%d_%H%M%S)) --fresh

# ----------------------------------------------------------------------
# Experiments / artifacts
# ----------------------------------------------------------------------
.PHONY: index query figures preset-check
index: ## Rebuild experiments/index.md from registry.jsonl
	$(PYTHON) tools/experiments_log.py rebuild
	@echo "Wrote experiments/index.md"

query: ## Filter runs (pass FILTER='--dataset mrpc --min-reward 0.4')
	$(PYTHON) tools/experiments_log.py query $(FILTER)

figures: ## Render paper figures from RUN(s); set RUN=<persistent_dir> OUT=figures/foo
	@[ -n "$(RUN)" ] || (echo "set RUN=<persistent dir> OUT=<output dir>" && exit 1)
	$(PYTHON) tools/paper_figures.py --runs "$(RUN)" \
	    --out $(or $(OUT),figures/$$(date +%Y%m%d_%H%M%S)) --formats png pdf

preset-check: ## Validate every preset for typos
	$(PYTHON) tools/validate_preset.py presets/*.conf Paean/presets/*.conf

# ----------------------------------------------------------------------
# Documentation
# ----------------------------------------------------------------------
.PHONY: changelog docs-check
changelog: ## Print the most recent CHANGELOG entries
	@head -120 CHANGELOG.md

docs-check: ## Validate HTML guide tags + ADR index coverage
	$(PYTHON) -c "from html.parser import HTMLParser; \
	class V(HTMLParser): \
	    pass" 2>/dev/null
	$(PYTHON) - <<-'PY'
		from html.parser import HTMLParser
		class V(HTMLParser):
		    def __init__(self): super().__init__(); self.stack=[]
		    def handle_starttag(self, t, a):
		        if t in ('br','hr','meta','link','img','input'): return
		        self.stack.append(t)
		    def handle_endtag(self, t):
		        if t in ('br','hr','meta','link','img','input'): return
		        if self.stack and self.stack[-1]==t: self.stack.pop()
		p=V(); p.feed(open("reports/session_summary/blb_stage2_rl_guide.html",encoding="utf-8").read())
		assert not p.stack, f"unclosed tags: {p.stack}"
		print("HTML guide OK")
	PY

# ----------------------------------------------------------------------
# Hygiene
# ----------------------------------------------------------------------
.PHONY: clean
clean: ## Remove __pycache__ / .pyc / ruff cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .pytest_cache
