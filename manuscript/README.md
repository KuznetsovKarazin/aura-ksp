# Manuscript source

This is the scientific manuscript prepared for the MDPI AI special issue on graph neural networks. It is an author manuscript, not an accepted journal publication. Author conflict-of-interest and funder-role declarations remain to be confirmed before submission.

Open `main.pdf` and `supplement.pdf`. For Overleaf, upload this directory and select `main.tex` (pdfLaTeX). For the supplement select `supplement.tex`. A complete TeX Live or MiKTeX installation can build locally:

```bash
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

The publisher's supplied `Definitions/` files retain their original contents; two PDF logo conversions are provided for portability. The paper and its original figures are available under CC BY 4.0; this does not relicense the research code, benchmark, or third-party template files. See `../LICENSE_SCOPE.md`.
