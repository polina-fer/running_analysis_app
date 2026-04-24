# Running Form Analyzer

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Make sure `side_view_models.joblib` is in this folder.

3. Run the app:
```
uvicorn main:app --reload
```

4. Open your browser at: http://localhost:8000

## Project structure
```
running_app/
├── main.py                     ← FastAPI backend
├── analyzer.py                 ← Your analysis pipeline (from notebook)
├── side_view_models.joblib     ← Your trained models
├── requirements.txt
└── static/
    └── index.html              ← Frontend
```
