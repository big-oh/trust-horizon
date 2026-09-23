# Regenerate every JSON result and every figure, then check the numbers quoted in the paper.
#   make            -> validation + experiments + ablation + figures + check
#   make paper      -> also rebuild paper/main_ieee.pdf (needs pdflatex and bibtex)
PY ?= python3

.PHONY: all data figures check paper clean

all: data figures check

data:
	$(PY) validate.py
	$(PY) experiments.py
	$(PY) ablation.py

figures:
	$(PY) figures.py

check:
	$(PY) paper_numbers.py

paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main_ieee && bibtex main_ieee \
	  && pdflatex -interaction=nonstopmode -halt-on-error main_ieee && pdflatex -interaction=nonstopmode -halt-on-error main_ieee

clean:
	rm -f validation.json results.json ablation.json abl_curves.json returns_pts.json
	rm -f paper/*.aux paper/*.log paper/*.blg paper/*.out
