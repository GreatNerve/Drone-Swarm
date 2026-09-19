.DEFAULT_GOAL := help
.PHONY: help test train sim sim-resume view view-pid view-train view-prod eval

UV ?= uv
VIEW ?= 0
TEST ?= 1
POLICY ?= baseline
CKPT ?= latest
MODE ?= prod
SEED ?= 0
RESUME ?= 0
SPEED ?= 4

RESUME_FLAG :=
ifeq ($(RESUME),1)
RESUME_FLAG := --resume
endif

help:
	@echo "swarm make targets"
	@echo ""
	@echo "  make help          this list"
	@echo "  make test          pytest"
	@echo "  make sim           train in Python 3D simulation (Ctrl+C saves latest.pt)"
	@echo "  make sim-resume    continue last sim train from models/latest.pt"
	@echo "  make train         headless train   VIEW=0 SEED=0  RESUME=1 to continue"
	@echo "  make view          play Test N        POLICY=baseline|rl CKPT=latest TEST=1"
	@echo "  make view-pid      PID baseline"
	@echo "  make view-train    Tests on models/latest.pt (in-progress train net)"
	@echo "  make view-prod     Tests on models/best.pt (prod / trained net)"
	@echo "  make eval          headless Tests 1-3, writes results/submit   MODE=prod|train|baseline"
	@echo ""
	@echo "models: models/latest.pt  (train dump)   models/best.pt  (prod dump)"
	@echo "submit: results/submit/   (images, graphs, result.md, data, copied ckpt)"
	@echo ""
	@echo "vars: SEED=$(SEED) TEST=$(TEST) POLICY=$(POLICY) CKPT=$(CKPT) MODE=$(MODE) VIEW=$(VIEW) SPEED=$(SPEED)"

test:
	$(UV) run pytest -q

train:
	$(UV) run python -m swarm train --view $(VIEW) --seed $(SEED) $(RESUME_FLAG)

sim:
	$(UV) run python -m swarm train --view python --seed $(SEED) $(RESUME_FLAG)

sim-resume:
	$(UV) run python -m swarm train --view python --seed $(SEED) --resume

view:
	$(UV) run python -m swarm view --policy $(POLICY) --ckpt $(CKPT) --test $(TEST) --seed $(SEED) --view python --speed $(SPEED)

view-pid:
	$(UV) run python -m swarm view --policy baseline --test $(TEST) --seed $(SEED) --view python --speed $(SPEED)

view-train:
	$(UV) run python -m swarm view --mode train --test $(TEST) --seed $(SEED) --view python --speed $(SPEED)

view-prod:
	$(UV) run python -m swarm view --mode prod --test $(TEST) --seed $(SEED) --view python --speed $(SPEED)

eval:
	$(UV) run python -m swarm eval --mode $(MODE) --seed $(SEED) --view $(VIEW)
