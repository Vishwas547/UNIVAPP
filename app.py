import os
from flask import Flask, render_template, request, flash, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- EMAIL CONFIG ----------------
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

if not SENDER_EMAIL or not APP_PASSWORD:
    raise RuntimeError("Missing EMAIL credentials")

# ---------------- TRAINING DATA ----------------
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

department_emails = {
    "Academic": "vishwasbekkanti@gmail.com",
    "Accounts": "accounts@university.edu",
    "Examination": "examcell@university.edu",
    "Scholarship": "scholarship@university.edu",
    "Hostel": "hosteloffice@university.edu"
}

def clean_text(text):
    ignore = ["respected sir","respected madam","thank you","regards"]
    text = text.lower()
    for i in ignore:
        text = text.replace(i,"")
    return text

# ---------------- EMAIL FUNCTION ----------------
def send_email(to_email, subject, body):

    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        name = request.form["name"]
        sid = request.form["sid"]
        dept = request.form["dept"]
        year = request.form["year"]
        req_text = request.form["request"]

        if not all([name,sid,dept,year,req_text]):
            flash("All fields required")
            return redirect("/")

        cleaned = clean_text(req_text)
        vec = vectorizer.transform([cleaned])
        predicted = model.predict(vec)[0]

        receiver = department_emails[predicted]

        body = f"""
Name: {name}
Student ID: {sid}
Department: {dept}
Year: {year}

------------------
Request:
{req_text}
"""

        try:
            send_email(receiver,
                       f"University Request - {predicted}",
                       body)

            flash(f"Request sent to {predicted} Department")

        except Exception as e:
            flash(str(e))

        return redirect("/")

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
