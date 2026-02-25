import os
from flask import Flask, render_template, request, flash, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dotenv import load_dotenv
from pymongo import MongoClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ---------------- LOAD ENV ----------------
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not all([SENDER_EMAIL, SENDGRID_API_KEY, MONGO_URI]):
    raise RuntimeError("Missing required env variables")

# ---------------- MONGO CONNECTION ----------------
client = MongoClient(MONGO_URI)
db = client["universityDB"]
requests_collection = db["requests"]

# ---------------- FLASK ----------------
app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- TRAIN DATA ----------------
requests_data = [
"leave permission","medical leave","attendance shortage","bonafide certificate",
"study certificate","transfer certificate","migration certificate",
"course completion certificate","internship permission letter",
"project approval academic","elective subject change",
"semester registration issue","section change request",
"academic calendar clarification","id card issue",

"tuition fee payment issue","fee receipt not generated","refund request",
"caution deposit refund","excess fee paid","fine payment issue",
"payment gateway failure","fee structure clarification",
"installment request","scholarship adjustment in fees",

"hall ticket not generated","wrong subject in hall ticket","exam fee issue",
"revaluation request","photocopy of answer script",
"supplementary exam registration","improvement exam",
"internal marks correction","grade card correction",
"backlog registration issue",

"scholarship not credited","nsp portal issue","minority scholarship",
"post matric scholarship","upload document correction",
"income certificate issue","bank account update",
"aadhaar mismatch","scholarship renewal problem",

"room allocation issue","room change request","water problem hostel",
"electricity issue hostel","wifi issue hostel",
"mess quality complaint","mess fee payment issue",
"furniture damage","cleaning issue hostel",
"security complaint hostel","gate pass permission",

"project topic approval","internship approval","lab permission",
"attendance condonation","faculty complaint",
"internal marks discussion","subject doubt clarification",
"research paper submission","recommendation letter",
"department event permission",

"serious grievance","faculty misconduct","harassment complaint",
"policy complaint","disciplinary issue",
"appeal against suspension","overall college complaint",

"placement registration issue","resume submission","internship opportunity",
"company drive details","offer letter issue",
"training program enrollment","aptitude training request",
"mock interview request","noc for internship",

"sports certificate","tournament participation",
"sports equipment issue","ground booking",
"sports quota certificate","attendance for sports",
"sports scholarship",

"library fine issue","book not available","lost book",
"library id issue","digital library access","thesis submission",

"bus pass issue","route change transport","bus timing issue",
"transport fee payment","new transport request",

"erp login issue","portal password reset",
"wifi campus issue","email id problem","software lab issue",

"ragging complaint","discrimination complaint",
"academic bias complaint"
]

departments = [
"Academic","Academic","Academic","Academic","Academic","Academic",
"Academic","Academic","Academic","Academic","Academic","Academic",
"Academic","Academic","Academic",

"Accounts","Accounts","Accounts","Accounts","Accounts",
"Accounts","Accounts","Accounts","Accounts","Accounts",

"Examination","Examination","Examination","Examination",
"Examination","Examination","Examination","Examination",
"Examination","Examination",

"Scholarship","Scholarship","Scholarship","Scholarship",
"Scholarship","Scholarship","Scholarship","Scholarship","Scholarship",

"Hostel","Hostel","Hostel","Hostel","Hostel",
"Hostel","Hostel","Hostel","Hostel","Hostel","Hostel",

"HOD","HOD","HOD","HOD","HOD",
"HOD","HOD","HOD","HOD","HOD",

"Principal","Principal","Principal","Principal",
"Principal","Principal","Principal",

"TPO","TPO","TPO","TPO","TPO",
"TPO","TPO","TPO","TPO",

"Sports","Sports","Sports","Sports",
"Sports","Sports","Sports",

"Library","Library","Library","Library","Library","Library",

"Transport","Transport","Transport","Transport","Transport",

"IT","IT","IT","IT","IT",

"Grievance","Grievance","Grievance"
]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(requests_data)

model = MultinomialNB()
model.fit(X, departments)

# ---------------- EMAIL MAP ----------------
department_emails = {
    "Academic": "academic@university.edu",
    "Accounts": "accounts@university.edu",
    "Examination": "examcell@university.edu",
    "Scholarship": "scholarship@university.edu",
    "Hostel": "hostel@university.edu",
    "HOD": "hod@university.edu",
    "Principal": "principal@university.edu",
    "TPO": "tpo@university.edu",
    "Sports": "sports@university.edu",
    "Library": "library@university.edu",
    "Transport": "transport@university.edu",
    "IT": "itsupport@university.edu",
    "Grievance": "grievance@university.edu"
}

# ---------------- EMAIL SEND ----------------
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

        vector = vectorizer.transform([req_text.lower()])
        probabilities = model.predict_proba(vector)[0]

        threshold = 0.15

        matched_departments = [
            model.classes_[i]
            for i, prob in enumerate(probabilities)
            if prob > threshold
        ]

        if not matched_departments:
            matched_departments = [model.predict(vector)[0]]

        receiver_emails = [
            department_emails[d]
            for d in matched_departments
        ]

        primary_department = matched_departments[0]

        # SAVE TO MONGO
        requests_collection.insert_one({
            "name": name,
            "student_id": sid,
            "student_email": student_email,
            "department": dept,
            "year": year,
            "predicted_departments": matched_departments,
            "request_text": req_text
        })

        try:

            department_body = f"""
Student Name: {name}
Student ID: {sid}
Student Email: {student_email}
Department: {dept}
Year: {year}

Request:
{req_text}

Routed To:
{', '.join(matched_departments)}
"""

            for email in receiver_emails:
                send_email(
                    email,
                    "University Request - AI Auto Routed",
                    department_body,
                    student_email
                )

            student_body = f"""
Dear {name},

Your request:

"{req_text}"

has been successfully received.

Please wait for 2 to 3 working days.

For further clarification contact:
{primary_department} Department

University Administration
"""

            send_email(
                student_email,
                "Request Received - Confirmation",
                student_body,
                SENDER_EMAIL
            )

            flash("Request submitted and routed successfully!")

        except Exception:
            flash("Saved to DB, but email failed.")

        return redirect("/")

    return render_template("index.html")

# ---------------- RUN ----------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
