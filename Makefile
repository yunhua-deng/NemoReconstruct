PYTHON ?= python3

.PHONY: backend-install backend-dev frontend-install frontend-dev openapi

backend-install:
	cd backend && $(PYTHON) -m pip install -r requirements.txt

backend-dev:
	cd backend && $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

openapi:
	cd backend && $(PYTHON) export_openapi.py
