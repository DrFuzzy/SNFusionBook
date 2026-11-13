# Build clean book; use make reviewed for both clean and red/blue comparison PDFs.
all: main.pdf

main.pdf: main.tex $(wildcard chapters/*.tex frontmatter/*.tex backmatter/*.tex bib/*.bib data/*/*.csv figures/*/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main

reviewed:
	python3 scripts/build_review_versions.py

examples:
	$(MAKE) -C companion_repo examples
	$(MAKE) -C companion_repo export

clean:
	latexmk -c main

.PHONY: all reviewed clean examples
