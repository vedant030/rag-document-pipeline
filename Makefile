.PHONY: ingest query demo test
ingest:
	python -m rag_pipeline ingest sample_docs
query:
	python -m rag_pipeline query "how long do you keep customer data?"
demo:
	streamlit run streamlit_app.py
test:
	pytest -q
