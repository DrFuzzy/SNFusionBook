# Companion repository of "Towards a Unified View of Sensor Fusion" (Springer Nature, 2026)
#
#   make install    install snfusion in editable mode (pip install -e .)
#   make test       run the unit and regression tests (pytest)
#   make examples   run every examples/*/ script (outputs go to each example's output/)
#   make notebooks  regenerate every .ipynb from the cell-annotated scripts (tools/py2nb.py)
#   make export     copy examples/*/output/*.csv -> ../data/<chNN>/ and *.pdf -> ../figures/<chNN>/
#   make clean      remove caches (generated outputs are kept under version control)

PYTHON ?= python3
BOOK_ROOT ?= ..

.PHONY: all install test examples notebooks export clean

all: test

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

examples:
	$(PYTHON) tools/run_all_examples.py --quiet

notebooks:
	@for f in $$(find examples exercises -mindepth 2 -maxdepth 2 -name '*.py' ! -name '_*' 2>/dev/null | sort); do \
		$(PYTHON) tools/py2nb.py $$f; \
	done

export:
	$(PYTHON) tools/export_outputs.py --book-root $(BOOK_ROOT)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache build *.egg-info snfusion.egg-info
