import os
from flask import Flask, render_template, request, flash, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dotenv import load_dotenv

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# -------------------------------------------------
# LOAD ENV VARIABLES
# -------------------------------------------------
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

if not SENDER_EMAIL or not SENDGRID_API_KEY:
    raise RuntimeError("Missing SENDGRID_API_KEY or SENDER_EMAIL")

# -------------------------------------------------
# FLASK SETUP
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "secret123"

# -------------------------------------------------
# TRAINING DATA
# -------------------------------------------------
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
    "Academic", "Academic",
    "Academic", "Academic",
    "Accounts", "Accounts",
    "Examination", "Examination",
    "Scholarship", "Scholarship",
    "Hostel", "Hostel"
]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X = vectorizer.fit_transform(requests_data)

model = MultinomialNB()
model.fit(X, departments)

# -------------------------------------------------
# DEPARTMENT EMAILS
# -------------------------------------------------
department_emails = {
    "Academic": "lnv4687@gmail.com",
    "Accounts": "accounts@university.edu",
    "Examination": "examcell@university.edu",
    "Scholarship": "scholarship@university.edu",
    "Hostel": "hosteloffice@university.edu"
}

# -------------------------------------------------
# CLEAN TEXT
# -------------------------------------------------
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

# -------------------------------------------------
# SEND EMAIL (SENDGRID)
# -------------------------------------------------
def send_email(to_email, subject, body):

    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body,
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print("SendGrid status:", response.status_code)

    except Exception as e:
        print("SENDGRID ERROR:", e)
        raise

# -------------------------------------------------
# ROUTE
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        name = request.form["name"]
        sid = request.form["sid"]
        dept = request.form["dept"]
        year = request.form["year"]
        req_text = request.form["request"]

        if not all([name, sid, dept, year, req_text]):
            flash("All fields are required")
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
            send_email(
                receiver,
                f"University Request - {predicted} Department",
                body,
            )

            flash(f"Request sent to {predicted} Department successfully!")

        except Exception:
            flash("Email sending failed. Check server logs.")

        return redirect("/")

    return render_template("index.html")


# -------------------------------------------------
# LOCAL RUN SUPPORT
# -------------------------------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
