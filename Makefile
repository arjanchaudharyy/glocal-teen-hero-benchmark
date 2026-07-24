.PHONY: help test eval tune cv ncv rank stats ask similar build serve
help:            ## show targets
	@grep -E '^[a-z].*:.*##' Makefile | sed 's/:.*##/ -/'
test:            ## run the unit tests (stdlib, zero deps)
	python3 -m unittest discover -s tests -v
eval:            ## retrieval evaluation: Recall / MRR / nDCG / MAP
	python3 -m gth eval
tune:            ## grid-search BM25 params on the gold set
	python3 -m gth tune
cv:              ## k-fold cross-validated config selection (proves generalization)
	python3 -m gth cv
ncv:             ## nested CV (hyperparams tuned in-fold, publish-safe estimate)
	python3 -m gth ncv
rank:            ## print the at-selection ranking
	python3 -m gth rank --top 20
stats:           ## cohort statistics by tier
	python3 -m gth stats
ask:             ## RAG query: make ask Q="who worked on climate?"
	python3 -m gth ask "$(Q)"
similar:         ## nearest honorees: make similar NAME="Rahul Ranjan Sah"
	python3 -m gth similar "$(NAME)"
build:           ## regenerate data/heroes.json + web corpus from the scored roster
	python3 build.py
serve:           ## serve the web app locally
	python3 -m http.server 8000
