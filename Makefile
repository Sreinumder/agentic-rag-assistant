.PHONY: install install-dev test lint fmt run backend ui eval docker

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff check --fix .
	ruff format .

run:
	python -m src.main

backend:
	uvicorn src.prod.backend:app --reload --port 8000

ui:
	streamlit run src.prod/ui.py

eval:
	python eval/harness.py --report

docker:
	docker compose up --build
