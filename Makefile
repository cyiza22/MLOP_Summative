.PHONY: help setup train run test docker clean

help:
	@echo "ML Pipeline - Available Commands"
	@echo "================================="
	@echo "make setup   - Setup environment"
	@echo "make train   - Train the model"
	@echo "make run     - Run the API"
	@echo "make docker  - Build and run with Docker"
	@echo "make test    - Run tests"
	@echo "make load    - Run load tests"
	@echo "make clean   - Clean up files"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

train:
	bash scripts/train_model.sh

run:
	python app.py

docker:
	docker-compose up --build

test:
	pytest tests/ -v

load:
	bash scripts/run_load_test.sh

clean:
	bash scripts/cleanup.sh