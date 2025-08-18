# Local CPU RAG Indexer

1. Install requirements:
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

2. Edit config.yaml: set DATA_PATH to your folder.

3. Run CLI:
   python cli.py --data_path ./data --reindex   # to build index
   python cli.py                                # interactive queries

4. Optional UI:
   streamlit run streamlit_app.py

Notes:
- To enable OCR, install tesseract on your OS.
- To enable video transcription, install ffmpeg and integrate Whisper.
- For large datasets, adjust FAISS_NLIST and FAISS_PQ_M in config.
