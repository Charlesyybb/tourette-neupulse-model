# 编译： make        （完整四遍 + bibtex）
#        make quick  （单遍，仅查语法错误）
#        make clean

MAIN = main
TEX  = xelatex -interaction=nonstopmode

all:
	$(TEX) $(MAIN).tex
	-bibtex $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex
	@echo "=== 完成： $(MAIN).pdf ==="

quick:
	$(TEX) -halt-on-error $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all quick clean
