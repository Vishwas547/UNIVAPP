import os
from flask import Flask, render_template, request, flash, redirect, url_for
import mysql.connector
from email.message import EmailMessage
import smtplib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dotenv import load_dotenv

# ---------------- Load Environment Variables ----------------
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------------- Database Connection ----------------
db = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    port=DB_PORT
)
cursor = db.cursor()

# ---------------- Email Function (STARTTLS) ----------------
def send_email(to_email, subject, body):
    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# ---------------- ML Model Setup ----------------
requests_data = [
    "requesting leave due to health issues",
    "medical leave application",
    "requesting bonafide certificate",
    "need bonafide for educational purpose",
    "fee paid but not updated",
    "scholarship amount not credited",
    "hall ticket not generated",
    "exam fee issue",
    "update phone number in scholarship site",
    "update personal details",
    "hostel room problem",
    "water problem in hostel"
]

departments = [
    "Academic", "Academic", "Academic", "Academic",
    "Accounts", "Accounts",
    "Examination", "Examination",
    "Scholarship", "Scholarship",
    "Hostel", "Hostel"
]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X = vectorizer.fit_transform(requests_data)

model = MultinomialNB()
model.fit(X, departments)

department_emails = {
    "Academic": "lnv4687@gmail.com",
    "Accounts": "accounts@university.edu",
    "Examination": "examcell@university.edu",
    "Scholarship": "scholarship@university.edu",
    "Hostel": "hosteloffice@university.edu"
}

def clean_text(text):
    ignore = [
        "respected sir",
        "respected madam",
        "thank you",
        "yours sincerely",
        "regards"
    ]
    text = text.lower()
    for word in ignore:
        text = text.replace(word, "")
    return text

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name")
        sid = request.form.get("sid")
        dept = request.form.get("dept")
        year = request.form.get("year")
        request_text = request.form.get("request")

        if not all([name, sid, dept, year, request_text]):
            flash("All fields are required", "danger")
            return redirect(url_for("index"))

        # Predict department
        cleaned = clean_text(request_text)
        vector = vectorizer.transform([cleaned])
        predicted_dept = model.predict(vector)[0]
        receiver_email = department_emails[predicted_dept]

        # Prepare email
        email_body = f"""
From:
Name: {name}
Student ID: {sid}
Department: {dept}
Class / Year: {year}

----------------------------------
Request:
{request_text}
"""

        # ---------------- Save to DB FIRST ----------------
        cursor.execute(
            """
            INSERT INTO requests
            (student_name, student_id, department,
             class_year, request_text,
             predicted_dept, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (name, sid, dept, year,
             request_text, predicted_dept, "Sent")
        )
        db.commit()

        # ---------------- Try sending email (safe) ----------------
        email_sent = True
        try:
            send_email(
                receiver_email,
                f"University Request - {predicted_dept} Department",
                email_body
            )
        except Exception as e:
            email_sent = False
            print("EMAIL ERROR:", e)

        if email_sent:
            flash(
                f"Request saved and sent to {predicted_dept} Department",
                "success"
            )
        else:
            flash(
                "Request saved but email could not be sent right now.",
                "warning"
            )

        return redirect(url_for("index"))

    return render_template("index.html")

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
