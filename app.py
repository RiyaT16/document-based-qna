from flask import Flask, request, jsonify, render_template
import tempfile
import backend_logic  # your backend logic module

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid file"}), 400

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        file.save(temp.name)
        file_path = temp.name

    try:
        backend_logic.ingest_pdf(file_path)
        return jsonify({"message": "Uploaded and indexed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["GET"])
def query():
    question = request.args.get("question")
    if not question:
        return jsonify({"error": "Missing question parameter"}), 400

    try:
        answer = backend_logic.answer_query(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8001)
