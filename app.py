import os
import requests
from flask import Flask, render_template, request, flash, redirect, url_for
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dotenv import load_dotenv

# ---------------- Load ENV ----------------
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

# ---------------- Database ----------------
db = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    port=DB_PORT,
)
cursor = db.cursor()

# ---------------- Brevo Email Sender ----------------
def send_email(to_email, subject, body):

    url = "https://api.brevo.com/v3/smtp/email"

    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "content-type": "application/json",
    }

    payload = {
        "sender": {
            "name": "University Request System",
            "email": FROM_EMAIL,
        },
        "to": [{"email": to_email}],
        "subject": subject,
        "textContent": body,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=15)

    if response.status_code not in (200, 201):
        raise Exception(response.text)


# ---------------- ML MODEL ----------------
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
    "water problem in hostel",
]

departments = [
    "Academic", "Academic", "Academic", "Academic",
    "Accounts", "Accounts",
    "Examination", "Examination",
    "Scholarship", "Scholarship",
    "Hostel", "Hostel",
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
    "Hostel": "hosteloffice@university.edu",
}

# ---------------- Utils ----------------
def clean_text(text):
    ignore = [
        "respected sir",
        "respected madam",
        "thank you",
        "yours sincerely",
        "regards",
    ]

    text = text.lower()
    for word in ignore:
        text = text.replace(word, "")
    return text


# ---------------- ROUTES ----------------
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

        cleaned = clean_text(request_text)
        vector = vectorizer.transform([cleaned])
        predicted_dept = model.predict(vector)[0]

        receiver_email = department_emails[predicted_dept]

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

        email_success = True

        try:
            send_email(
                receiver_email,
                f"University Request - {predicted_dept} Department",
                email_body,
            )

        except Exception as e:
            print("EMAIL ERROR:", e)
            email_success = False

        status = "Email Sent" if email_success else "Saved - Email Failed"

        cursor.execute(
            """
            INSERT INTO requests
            (student_name, student_id, department, class_year,
             request_text, predicted_dept, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                name,
                sid,
                dept,
                year,
                request_text,
                predicted_dept,
                status,
            ),
        )
        db.commit()

        if email_success:
            flash(f"Request sent to {predicted_dept} department", "success")
        else:
            flash("Saved to DB but email delivery failed.", "warning")

        return redirect(url_for("index"))

    return render_template("index.html")


# ---------------- Run Locally ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
