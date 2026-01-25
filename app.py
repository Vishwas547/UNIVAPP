from flask import Flask, render_template, request, redirect, url_for
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import pdfplumber
import re
from datetime import datetime

# ---------------- APP CONFIG ----------------

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- MONGODB ----------------

MONGO_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGO_URI)

db = client["resume_app"]
users_col = db["users"]
analysis_col = db["analysis"]

# ---------------- LOGIN ----------------

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, user):
        self.id = str(user["_id"])
        self.email = user["email"]

@login_manager.user_loader
def load_user(user_id):
    user = users_col.find_one({"_id": ObjectId(user_id)})
    return User(user) if user else None

# ---------------- LOGIC ----------------

SKILLS = [
    "python", "java", "c++", "sql", "javascript", "html", "css",
    "react", "node", "flask", "django",
    "machine learning", "data science",
    "aws", "docker", "git"
]

def clean_text(text):
    return re.sub(r"\s+", " ", text.lower())

def extract_skills(text):
    return [s for s in SKILLS if s in clean_text(text)]

def analyze_resume_with_jd(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    overlap = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    score = int((len(overlap) / len(jd_skills)) * 100) if jd_skills else 0

    tips = []
    if missing:
        tips.append("Add missing JD skills if applicable.")
    if "project" not in resume_text.lower():
        tips.append("Include a Projects section.")
    if "intern" not in resume_text.lower():
        tips.append("Mention internships.")

    return score, overlap, missing, tips

# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    score = None
    overlap = []
    missing = []
    tips = []

    if request.method == "POST":
        file = request.files.get("resume")
        jd_text = request.form.get("job_description", "")

        if file:
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

            score, overlap, missing, tips = analyze_resume_with_jd(text, jd_text)

            analysis_col.insert_one({
                "user_id": current_user.id,
                "score": score,
                "created_at": datetime.utcnow()
            })

    return render_template(
        "index.html",
        score=score,
        overlap=overlap,
        missing=missing,
        tips=tips
    )

@app.route("/history")
@login_required
def history():
    records = list(
        analysis_col.find({"user_id": current_user.id})
        .sort("created_at", -1)
    )

    scores = [r["score"] for r in records]
    dates = [r["created_at"].strftime("%d %b") for r in records]

    return render_template(
        "history.html",
        analyses=records,
        scores=scores,
        dates=dates
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        if users_col.find_one({"email": email}):
            return "User already exists"

        user_id = users_col.insert_one({
            "email": email,
            "password": password
        }).inserted_id

        login_user(User(users_col.find_one({"_id": user_id})))
        return redirect(url_for("index"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = users_col.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            login_user(User(user))
            return redirect(url_for("index"))

        return "Invalid credentials"

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run()
