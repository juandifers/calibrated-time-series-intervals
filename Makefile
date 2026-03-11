PYTHON ?= python
BASE_URL ?= http://127.0.0.1:8000

.PHONY: install fetch-model run-api smoke validate-artifact render-notebooks

install:
	$(PYTHON) -m pip install -r requirements.txt

fetch-model:
	$(PYTHON) scripts/fetch_demo_model.py

run-api:
	$(PYTHON) scripts/run_demo_api.py

smoke:
	$(PYTHON) scripts/smoke_test_demo_api.py --base-url $(BASE_URL)

validate-artifact:
	$(PYTHON) scripts/validate_demo_artifact.py

render-notebooks:
	$(PYTHON) scripts/render_public_notebooks.py
