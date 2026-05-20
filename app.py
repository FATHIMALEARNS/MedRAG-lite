from flask import request, jsonify, make_response
from rag_pipeline import run_rag_pipeline
import os
import sqlite3
from flask import Flask, render_template, request, redirect, session, url_for
from functools import wraps
from werkzeug.utils import secure_filename

from scripts.auth import init_auth_db, login_user, register_user
from ml.predict import predict   # 🔥 REAL MODEL
from models import db, AnalysisRecord

app = Flask(__name__)
app.secret_key = "mediclear-secret-key"

# ---------------- DATABASE CONFIG ----------------
basedir = os.path.abspath(os.path.dirname(__file__))
# Note: we use medrag.db to sit alongside the raw SQL DB if they share the same file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'medrag.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DB_PATH = "medrag.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_auth_db()

# ---------------- HELPERS ----------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = login_user(
            request.form.get("email"),
            request.form.get("password")
        )
        if user:
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        # In a real app, this would send an email via SMTP.
        # For a local offline project, we simply mock the success response.
        return render_template("forgot_password.html", success="If an account matches that email, a reset secure link has been sent. Please check your inbox.")
    return render_template("forgot_password.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if register_user(
            request.form.get("name"),
            request.form.get("email"),
            request.form.get("password")
        ):
            return redirect(url_for("login"))
        return render_template("register.html", error="Email already exists")
    return render_template("register.html")

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        file = request.files.get("image")
        user_query = request.form.get("user_query") or "What does this chest X-ray result mean?"
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            image_path = f"/static/uploads/{filename}"

            session["uploaded_image"] = image_path
            session["user_query"] = user_query
            session["patient_symptoms"] = request.form.get("patient_symptoms", "")
            # Clear previous result session vars to ensure fresh state
            session.pop("confidence", None)
            session.pop("label", None)
            session.pop("probability", None)

            return redirect(url_for("result"))

    return render_template("dashboard.html", user_name=session.get("user_name"))

@app.route("/processing")
@login_required
def processing():
    return render_template("processing.html")

@app.route("/result")
@login_required
def result():
    return render_template(
        "result.html",
        user_name=session.get("user_name"),
        image_path=session.get("uploaded_image"),
        user_query=session.get("user_query"),
        patient_symptoms=session.get("patient_symptoms", "")
    )

@app.route("/history")
@login_required
def history():
    records = AnalysisRecord.query.filter_by(user_id=session["user_id"]).order_by(AnalysisRecord.timestamp.desc()).all()
    return render_template("history.html", history=[r.to_dict() for r in records])

@app.route("/record/<int:record_id>")
@login_required
def view_record(record_id):
    record = AnalysisRecord.query.filter_by(id=record_id, user_id=session["user_id"]).first_or_404()
    return render_template("record.html", record=record.to_dict())

@app.route("/export_pdf/<int:record_id>")
@login_required
def export_pdf(record_id):
    record = AnalysisRecord.query.filter_by(id=record_id, user_id=session["user_id"]).first_or_404()
    base_url = request.host_url
    return render_template("pdf_report.html", record=record.to_dict(), doc_name=session.get("user_name"), base_url=base_url)

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user_name=session.get("user_name"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    API endpoint to run MedRAG pipeline
    """

    data = request.json

    image_path = data.get("image_path")
    user_query = data.get("user_query")
    patient_symptoms = data.get("patient_symptoms", "")

    if not image_path or not user_query:
        return jsonify({"error": "image_path and user_query are required"}), 400

    result = run_rag_pipeline(image_path, user_query, patient_symptoms)

    # 🚨 Check if validator returned error
    if "error" in result:
        return jsonify(result), 400

    # Save to history via SQLAlchemy
    ml_res = result["ml_result"]
    new_record = AnalysisRecord(
        user_id=session.get("user_id"),
        patient_name="Patient " + str(session.get("user_id", "Unknown")),
        image_path=image_path,
        patient_symptoms=patient_symptoms,
        user_query=user_query,
        cnn_label=ml_res.get("label"),
        cnn_confidence=ml_res.get("confidence"),
        cnn_probability=ml_res.get("probability"),
        llm_report=result["final_answer"]
    )
    db.session.add(new_record)
    db.session.commit()

    return jsonify({
        "prediction": ml_res,
        "response": result["final_answer"],
        "record_id": new_record.id
    })

@app.route("/analyze-ui")
def analyze_ui():
    return render_template("analyze.html")

if __name__ == "__main__":
    app.run(debug=True)
