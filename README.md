# Intelligent Web Application Compiler for Detecting JavaScript Errors Using Machine Learning

## How it works
1. **Compiler stage** - Esprima parses the code and reports syntax errors with line/column.
2. **Static analysis** - AST scope analysis finds undefined variables, const reassignment, unused variables,
   assignment in conditions, NaN comparison, eval().
3. **Machine learning** - a TF-IDF (character n-gram) + Logistic Regression model classifies each line as
   `ok`, `syntax`, `type_error`, `logic_error` or `bad_practice` with a confidence score.
4. Results are merged, de-duplicated and shown with a health score.

## Run
```
pip install -r requirements.txt
python train_model.py     # trains and saves model.joblib (a trained model is already included)
python app.py             # open http://127.0.0.1:5000
```

## Files
- `app.py` Flask server and `/api/analyze` endpoint
- `analyzer.py` parser + rules + ML inference
- `train_model.py` dataset generator and model training
- `static/index.html` web interface
