PYTHON ?= python3

.PHONY: backend-install backend-dev frontend-install frontend-dev openapi

backend-install:
	cd backend && $(PYTHON) -m pip install -r requirements.txt

backend-dev:
	cd backend && uvicorn app.main:app --reload

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

openapi:
	cd backend && $(PYTHON) export_openapi.py
