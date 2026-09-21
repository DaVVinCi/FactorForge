.PHONY: install demo test app

install:
	python -m pip install -e ".[ui,dev]"

demo:
	python -m factorforge demo --output-dir artifacts/demo

test:
	python -m pytest

app:
	streamlit run streamlit_app.py

