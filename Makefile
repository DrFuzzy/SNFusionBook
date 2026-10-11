# Build the book (requires a TeX Live distribution with pdflatex, bibtex, makeindex).
#   make          -> main.pdf (with the revision markup, blue = added, red = removed)
#   make clean    -> remove auxiliary files
#   make examples -> regenerate data/ and generated figures from companion_repo (optional,
#                    NOT needed to compile: the generated CSV/PDF files are shipped in data/ and figures/)
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

all: main.pdf

main.pdf: main.tex chapters/*.tex frontmatter/*.tex backmatter/*.tex bib/references.bib
	$(PDFLATEX) main
	bibtex main
	makeindex -s style/svind.ist main
	$(PDFLATEX) main
	$(PDFLATEX) main

examples:
	$(MAKE) -C companion_repo export

clean:
	rm -f main.aux main.bbl main.blg main.idx main.ilg main.ind main.log main.out main.toc main.lof main.lot main.synctex.gz
	rm -f chapters/*.aux frontmatter/*.aux backmatter/*.aux

.PHONY: all clean examples
