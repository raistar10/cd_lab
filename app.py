import os
from flask import Flask, request, jsonify, send_from_directory
import analyzer

app = Flask(__name__, static_folder="static")

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.post("/api/analyze")
def api_analyze():
    code = (request.get_json(silent=True) or {}).get("code", "")
    if not code.strip(): return jsonify(error="Please paste some JavaScript code."), 400
    if len(code) > 200_000: return jsonify(error="Code is too large (200 KB max)."), 413
    return jsonify(analyzer.analyze(code))

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
