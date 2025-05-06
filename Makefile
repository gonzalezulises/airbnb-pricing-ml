install:
	pip install -r requirements.txt

notebook:
	jupyter lab

train:
	python src/train_model.py

lint:
	ruff check .

clean:
	find . -name '__pycache__' -exec rm -rf {} +
