import os
from flask import Flask, render_template, request, flash, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dotenv import load_dotenv
import psycopg2

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ---------------- LOAD ENV ----------------
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not all([SENDER_EMAIL, SENDGRID_API_KEY, DATABASE_URL]):
    raise RuntimeError("Missing required env variables")

# ---------------- FLASK ----------------
app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- DB CONNECTION ----------------
def get_db():
    return psycopg2.connect(DATABASE_URL)

# ---------------- TRAIN DATA ----------------
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
    "Academic","Academic",
    "Academic","Academic",
    "Accounts","Accounts",
    "Examination","Examination",
    "Scholarship","Scholarship",
    "Hostel","Hostel"
]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(requests_data)

model = MultinomialNB()
model.fit(X, departments)

# ---------------- EMAILS ----------------
department_emails = {
    "Academic": "vishwasbekkanti@gmail.com",
    "Accounts": "accounts@university.edu",
    "Examination": "examcell@university.edu",
    "Scholarship": "scholarship@university.edu",
    "Hostel": "hosteloffice@university.edu"
}

# ---------------- CLEAN ----------------
def clean_text(text):
    ignore = [
        "respected sir",
        "respected madam",
        "thank you",
        "regards",
        "yours sincerely",
    ]
    text = text.lower()
    for i in ignore:
        text = text.replace(i, "")
    return text

# ---------------- SEND EMAIL ----------------
def send_email(to_email, subject, body, reply_to):

    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body,
    )

    message.reply_to = reply_to

    sg = SendGridAPIClient(SENDGRID_API_KEY)
    sg.send(message)

# ---------------- CREATE TABLE IF NOT EXISTS ----------------
def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id SERIAL PRIMARY KEY,
        name TEXT,
        student_id TEXT,
        student_email TEXT,
        department TEXT,
        year TEXT,
        predicted_department TEXT,
        request_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

init_db()

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        name = request.form["name"]
        sid = request.form["sid"]
        student_email = request.form["email"]
        dept = request.form["dept"]
        year = request.form["year"]
        req_text = request.form["request"]

        if not all([name, sid, student_email, dept, year, req_text]):
            flash("All fields are required")
            return redirect("/")

        cleaned = clean_text(req_text)
        vec = vectorizer.transform([cleaned])
        predicted = model.predict(vec)[0]

        receiver = department_emails[predicted]

        # -------- SAVE TO DB --------
        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO requests
        (name, student_id, student_email, department, year,
         predicted_department, request_text)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (name, sid, student_email, dept, year, predicted, req_text))

        conn.commit()
        cur.close()
        conn.close()

        # -------- SEND EMAIL --------
        try:
            send_email(
                receiver,
                f"Student Request: {predicted} Department",
                f"""
Name: {name}
Student ID: {sid}
Student Email: {student_email}
Department: {dept}
Year: {year}

------------------
Request:
{req_text}
""",
                student_email,
            )

            flash("Request submitted and saved successfully!")

        except Exception:
            flash("Saved to DB, but email failed.")

        return redirect("/")

    return render_template("index.html")

# ---------------- LOCAL RUN ----------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
