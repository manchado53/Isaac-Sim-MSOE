# Build the WACV paper PDF.  Usage: make   (or: make clean)
paper.pdf: paper.tex paper.bib cvpr.sty ieeenat_fullname.bst
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz *.brf

.PHONY: clean
