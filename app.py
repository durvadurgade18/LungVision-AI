import os
import numpy as np
import tensorflow as tf
import sqlite3
import random
import smtplib
from gtts import gTTS
import io
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image as RLImage
)

NAVY = colors.HexColor("#1a2f4f")
BLUE = colors.HexColor("#2563eb")
GREEN = colors.HexColor("#16a34a")
RED = colors.HexColor("#dc2626")
GRAY_TEXT = colors.HexColor("#6b7280")
BORDER = colors.HexColor("#e5e7eb")
# Minimum confidence required to trust a disease prediction
MIN_CONFIDENCE = 20.0   # since your confidence is already in % (0-100), not 0-1
from email.mime.text import MIMEText
from datetime import datetime, timedelta

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    flash,
    send_file          # NEW: needed to send the generated PDF back to the browser
)

from flask_cors import CORS
from dotenv import load_dotenv

from PIL import Image

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)
from datetime import datetime
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

import clip
import torch

# NEW: PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# Local imports

from chatbot import (
    ask_gemini,
    is_hospital_query
)

from medical import (
    get_nearby_hospitals
)

from database.database import get_db_connection



# ===============================
# Environment
# ===============================


load_dotenv()

EMAIL = os.getenv("GMAIL_EMAIL")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


def send_email_otp(receiver_email, otp):

    subject = "LungVision AI Password Reset OTP"

    body = f"""
Hello,

Your OTP for resetting your LungVision AI password is:

{otp}

This OTP is valid for 5 minutes.

If you did not request this, please ignore this email.

Regards,
LungVision AI
"""

    msg = MIMEText(body)

    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = receiver_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL, APP_PASSWORD)
        server.send_message(msg)


otp_storage = {}
OTP_EXPIRY_MINUTES = 5
# ===============================
# Flask Setup
# ===============================

app = Flask(__name__)
CORS(app)
app.secret_key = "lungvision_secret_key"
DATABASE = "database/lungvision.db"



def init_db():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
CREATE TABLE IF NOT EXISTS appointments(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    patient_name TEXT,
    phone TEXT,
    disease TEXT,
    doctor TEXT,
    appointment_date TEXT,
    appointment_time TEXT,
    symptoms TEXT,
    created_at TEXT,
    report_filename TEXT

)
""")

    conn.commit()
    conn.close()

init_db()






# ===============================
# Frontend Routes
# ===============================


@app.route("/")
def root():

    return redirect(
        url_for("login_page")
    )






@app.route("/home")
def home():

    return render_template(
        "index.html"
    )



@app.route("/predict")
def predict_page():

    return render_template(
        "predict.html"
    )
@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():

    try:

        image_path = session.get("last_xray_path")
        disease = session.get("last_prediction")
        confidence = session.get("last_confidence")
        target_index = session.get("last_class_index")

        if not image_path:
            return jsonify({
                "error": "No analyzed X-ray found."
            }), 400

        if not disease or target_index is None:
            return jsonify({
                "error": "No prediction data found. Analyze the X-ray again."
            }), 400

        print("🔥 Generating Grad-CAM for:", image_path)
        print("🔥 Disease:", disease)
        print("🔥 Confidence:", confidence)
        print("🔥 Target class:", target_index)

        result = generate_gradcam(
            image_path,
            target_index
        )

        return jsonify({
            "success": True,
            "disease": disease,
            "confidence": confidence,
            "original": result["original"],
            "heatmap": result["heatmap"],
            "overlay": result["overlay"]
        })

    except Exception as e:

        print("❌ Grad-CAM Error:", e)

        return jsonify({
            "error": str(e)
        }), 500
    
@app.route("/heatmap")
def heatmap_page():

    disease = session.get("last_prediction")
    confidence = session.get("last_confidence")

    print("🔥 Opening heatmap page")
    print("🔥 Disease:", disease)
    print("🔥 Confidence:", confidence)

    if disease is None:
        return redirect(url_for("predict_page"))

    return render_template(
        "heatmap.html",
        disease=disease,
        confidence=confidence
    )


@app.route("/about")
def about():

    return render_template(
        "about.html"
    )



@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )



@app.route("/assistant")
def assistant_page():

    return render_template(
        "assistant.html"
    )



@app.route("/specialists")
def specialists():

    return render_template(
        "specialists.html"
    )



@app.route("/appointment", methods=["GET", "POST"])
def appointment():

    if request.method == "POST":

        patient_name = request.form.get("patient_name") 
        phone = request.form.get("phone")
        disease = request.form.get("disease")
        doctor = request.form.get("doctor")
        appointment_date = request.form.get("appointment_date")
        appointment_time = request.form.get("appointment_time")
        symptoms = request.form.get("symptoms")
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---- NEW: Handle uploaded PDF report ----
        report_filename = None
        report_file = request.files.get("report")

        if report_file and report_file.filename != "":
            os.makedirs("uploads/reports", exist_ok=True)
            report_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_file.filename}"
            report_file.save(os.path.join("uploads/reports", report_filename))

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO appointments
(
    patient_name,
    phone,
    disease,
    doctor,
    appointment_date,
    appointment_time,
    symptoms,
    created_at,
    report_filename
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
    patient_name,
    phone,
    disease,
    doctor,
    appointment_date,
    appointment_time,
    symptoms,
    created_at,
    report_filename
))

        conn.commit()
        conn.close()

        return redirect(url_for("appointment_success"))

    return render_template("appointment.html")



@app.route("/appointment-success")
def appointment_success():

    return render_template(
        "appointment-success.html"
    )

@app.route("/admin/appointments")
def admin_appointments():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointments ORDER BY id DESC")

    appointments = cursor.fetchall()

    conn.close()

    return render_template(
        "admin_appointments.html",
        appointments=appointments
    )
# ===============================
# Contact Form
# ===============================


@app.route("/submit_contact", methods=["POST"])
def submit_contact():

    name = request.form.get("name")
    email = request.form.get("email")
    phone = request.form.get("phone")
    subject = request.form.get("subject")
    message = request.form.get("message")


    conn = get_db_connection()

    cursor = conn.cursor()


    cursor.execute("""
        INSERT INTO contact_messages
        (
            full_name,
            email,
            phone,
            subject,
            message
        )
        VALUES (?, ?, ?, ?, ?)
    """,
    (
        name,
        email,
        phone,
        subject,
        message
    ))


    conn.commit()
    conn.close()


    flash(
        "Message sent successfully!",
        "success"
    )


    return redirect(
        url_for("contact")
    )



# ===============================
# Login Routes
# ===============================


@app.route("/login", methods=["GET"])
def login_page():

    return render_template(
        "login.html"
    )



@app.route("/login", methods=["POST"])
def login_user():

    email = request.form.get("email")
    password = request.form.get("password")


    conn = get_db_connection()

    cursor = conn.cursor()


    cursor.execute(
        "SELECT * FROM users WHERE email=?",
        (email,)
    )


    user = cursor.fetchone()


    conn.close()



    if user and check_password_hash(
        user["password_hash"],
        password
    ):


        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["user_email"] = user["email"]


        return redirect(
            url_for("home")
        )



    return render_template(
        "login.html",
        error="Invalid Email or Password"
    )



@app.route("/logout")
def logout():

    session.clear()


    return redirect(
        url_for("login_page")
    )
# ===============================
# Forgot Password
# ===============================

@app.route("/forgot-password", methods=["GET"])
def forgot_password():

    return render_template("forgot_password.html")


# ===============================
# Send OTP
# ===============================

@app.route("/send-otp", methods=["POST"])
def send_otp():

    email = request.form.get("email")
    print("Email entered:", email)

    if not email:

        flash("Please enter your email address.", "danger")

        return redirect(url_for("forgot_password"))

    # Check if phone number exists
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
    "SELECT * FROM users WHERE email = ?",
    (email,)
)

    user = cursor.fetchone()
    print("User found:", user)

    conn.close()

    if not user:

        flash("Phone number is not registered.", "danger")

        return redirect(url_for("forgot_password"))

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))

    # Store OTP in memory
    otp_storage[email] = {
        "otp": otp,
        "expires": datetime.now() + timedelta(minutes=5)
    }

    try:

        send_email_otp(email, otp)

        session["reset_email"] = email

        flash("OTP sent successfully!", "success")
        return redirect(url_for("verify_otp_page"))

    except Exception as e:

        print("Twilio Error:", e)

        flash("Failed to send OTP.", "danger")

        return redirect(url_for("forgot_password"))

# ===============================
# Verify OTP Page
# ===============================

@app.route("/verify-otp", methods=["GET"])
def verify_otp_page():

    return render_template("verify_otp.html")


# ===============================
# Verify OTP
# ===============================

@app.route("/verify-otp", methods=["POST"])
def verify_otp():

    otp = request.form.get("otp")

    email = session.get("reset_email")

    if not email:

        flash("Session expired.", "danger")

        return redirect(url_for("forgot_password"))

    if email not in otp_storage:

        flash("OTP expired.", "danger")

        return redirect(url_for("forgot_password"))

    stored = otp_storage[email]

    if datetime.now() > stored["expires"]:

        otp_storage.pop(phone)

        flash("OTP expired.", "danger")

        return redirect(url_for("forgot_password"))

    if otp != stored["otp"]:

        flash("Invalid OTP.", "danger")

        return redirect(url_for("verify_otp_page"))

    otp_storage.pop(email)

    session["otp_verified"] = True

    return redirect(url_for("reset_password_page"))

# ===============================
# Reset Password Page
# ===============================

@app.route("/reset-password", methods=["GET"])
def reset_password_page():

    if not session.get("otp_verified"):
        return redirect(url_for("forgot_password"))

    return render_template("reset_password.html")


# ===============================
# Update Password
# ===============================

@app.route("/reset-password", methods=["POST"])
def reset_password():

    if not session.get("otp_verified"):
        return redirect(url_for("forgot_password"))

    password = request.form.get("password")
    confirm_password = request.form.get("confirm_password")

    if password != confirm_password:

        flash("Passwords do not match.", "danger")
        return redirect(url_for("reset_password_page"))

    password_hash = generate_password_hash(password)

    email = session.get("reset_email")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
    """
    UPDATE users
    SET password_hash=?
    WHERE email=?
    """,
    (password_hash, email)
)

    conn.commit()
    conn.close()

    session.pop("otp_verified", None)
    session.pop("reset_email", None)

    flash("Password updated successfully.", "success")

    return redirect(url_for("login_page"))

# ===============================
# Register
# ===============================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "GET":

        return render_template(
            "register.html"
        )

    fullname = request.form.get("fullname")

    email = request.form.get("email")

    print("Register Email:", email)

    phone = request.form.get("phone")

    password = request.form.get("password")

    confirm_password = request.form.get("confirm_password")


    if password != confirm_password:

        return "Passwords do not match"


    password_hash = generate_password_hash(
        password
    )


    conn = get_db_connection()

    cursor = conn.cursor()


    try:

        cursor.execute("""
            INSERT INTO users
            (
                full_name,
                email,
                phone,
                password_hash
            )

            VALUES (?, ?, ?, ?)

        """,
        (
            fullname,
            email,
            phone,
            password_hash
        ))

    except sqlite3.IntegrityError:

        conn.close()

        return render_template(
            "register.html",
            error="Email already registered. Please use another email or login."
        )


    conn.commit()

    conn.close()


    return redirect(
        url_for("login_page")
    )

# ===============================
# Load X-Ray Validator Model
# ===============================

VALIDATOR_MODEL_PATH = "models/xray_validator_final.keras"

validator_model = tf.keras.models.load_model(
    VALIDATOR_MODEL_PATH,
    compile=False
)

print("X-Ray Validator Loaded Successfully")


# ===============================
# Load CLIP Model (extra validation layer)
# ===============================

clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)

clip_text_labels = [
    "a chest x-ray showing ribs and lungs",
    "an x-ray of the leg, arm, hand, knee, or another body part that is not the chest",
    "a screenshot of a website or app",
    "a photo of nature, people, or objects"
]

clip_text_tokens = clip.tokenize(clip_text_labels).to(clip_device)

print("CLIP Model Loaded Successfully")


# ===============================
# Load Lung Disease Model
# ===============================


MODEL_PATH = "models/best_model.keras"



model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)



print("Model Loaded Successfully")



# Same order as training

class_names = [

    "Atelectasis",

    "COVID19",

    "Emphysema",

    "Lung_Cancer",

    "Normal",

    "Pulmonary_Edema",

    "Pulmonary_Fibrosis",

    "Tuberculosis"

]


# ===============================
# Home Care Precautions (multi-language)
# ===============================

PRECAUTIONS = {

    "Atelectasis": {
        "en": ["Practice deep breathing exercises regularly", "Sit upright rather than lying flat", "Use an incentive spirometer if prescribed", "Avoid smoking and secondhand smoke", "Stay physically active as tolerated", "Seek emergency care if breathlessness worsens suddenly"],
        "hi": ["नियमित रूप से गहरी सांस लेने का अभ्यास करें", "लेटने के बजाय सीधे बैठें", "यदि डॉक्टर ने सलाह दी हो तो इंसेंटिव स्पाइरोमीटर का उपयोग करें", "धूम्रपान और सेकेंडहैंड धुएं से बचें", "जितना सहन कर सकें उतना सक्रिय रहें", "सांस लेने में अचानक कठिनाई बढ़े तो आपातकालीन सहायता लें"],
        "mr": ["नियमितपणे दीर्घ श्वासोच्छवासाचा व्यायाम करा", "झोपण्याऐवजी सरळ बसा", "डॉक्टरांनी सांगितल्यास इन्सेंटिव्ह स्पायरोमीटर वापरा", "धूम्रपान आणि इतरांच्या धुरापासून दूर राहा", "सहन होईल तितके सक्रिय राहा", "श्वास घेण्यास त्रास वाढल्यास तातडीने मदत घ्या"],
    },

    "COVID19": {
        "en": ["Isolate yourself to avoid spreading infection", "Rest adequately and stay hydrated", "Monitor oxygen levels with a pulse oximeter if available", "Sit upright to ease breathing difficulty", "Maintain good ventilation in your room", "Seek emergency care immediately if breathing becomes severely difficult"],
        "hi": ["संक्रमण फैलने से बचने के लिए स्वयं को अलग रखें", "पर्याप्त आराम करें और पानी पीते रहें", "यदि उपलब्ध हो तो पल्स ऑक्सीमीटर से ऑक्सीजन स्तर जांचें", "सांस लेने में आसानी के लिए सीधे बैठें", "कमरे में हवा का अच्छा आवागमन बनाए रखें", "सांस लेना कठिन हो जाए तो तुरंत सहायता लें"],
        "mr": ["संसर्ग पसरू नये म्हणून स्वतःला वेगळे ठेवा", "पुरेशी विश्रांती घ्या आणि पाणी पीत राहा", "उपलब्ध असल्यास पल्स ऑक्सिमीटरने तपासा", "श्वास सोपे जाण्यासाठी सरळ बसा", "खोलीत हवेचा योग्य प्रवाह ठेवा", "श्वास घेण्यास त्रास झाल्यास त्वरित मदत घ्या"],
    },

    "Emphysema": {
        "en": ["Avoid smoking and exposure to air pollution", "Practice pursed-lip breathing techniques", "Engage in light physical activity as advised", "Avoid cold air and sudden temperature changes", "Keep indoor air clean and well ventilated", "Seek emergency care if breathlessness worsens severely"],
        "hi": ["धूम्रपान और वायु प्रदूषण से बचें", "पर्स्ड-लिप ब्रीदिंग तकनीक का अभ्यास करें", "सलाह अनुसार हल्की शारीरिक गतिविधि करें", "ठंडी हवा और तापमान में बदलाव से बचें", "घर की हवा को स्वच्छ रखें", "गंभीर कठिनाई होने पर सहायता लें"],
        "mr": ["धूम्रपान आणि वायू प्रदूषण टाळा", "पर्स्ड-लिप श्वासोच्छवास तंत्राचा सराव करा", "सल्ल्यानुसार सौम्य हालचाल करा", "थंड हवा व तापमान बदल टाळा", "घरातील हवा स्वच्छ ठेवा", "गंभीर त्रास झाल्यास मदत घ्या"],
    },

    "Lung_Cancer": {
        "en": ["Avoid smoking and secondhand smoke completely", "Eat a nutritious, well-balanced diet", "Rest adequately and manage fatigue", "Attend all follow-up medical appointments", "Avoid exposure to industrial dust and chemicals", "Seek emergency care for unexplained severe pain or breathlessness"],
        "hi": ["धूम्रपान और धुएं से पूरी तरह बचें", "पौष्टिक आहार लें", "पर्याप्त आराम करें", "सभी फॉलो-अप अपॉइंटमेंट में जाएं", "औद्योगिक धूल-रसायनों से बचें", "गंभीर दर्द या सांस फूलने पर सहायता लें"],
        "mr": ["धूम्रपान आणि धुरापासून दूर राहा", "पौष्टिक आहार घ्या", "पुरेशी विश्रांती घ्या", "सर्व फॉलो-अप भेटींना जा", "औद्योगिक धूळ-रसायने टाळा", "तीव्र वेदना/त्रास झाल्यास मदत घ्या"],
    },

    "Normal": {
        "en": ["No disease detected in this scan", "Maintain a healthy, active lifestyle", "Avoid smoking and polluted environments", "Continue regular health checkups", "Eat a balanced diet and stay hydrated", "Consult a doctor if you experience any new symptoms"],
        "hi": ["इस स्कैन में कोई बीमारी नहीं पाई गई", "स्वस्थ जीवनशैली बनाए रखें", "धूम्रपान और प्रदूषण से बचें", "नियमित जांच जारी रखें", "संतुलित आहार लें", "नया लक्षण दिखे तो डॉक्टर से सलाह लें"],
        "mr": ["या स्कॅनमध्ये आजार आढळला नाही", "निरोगी जीवनशैली राखा", "धूम्रपान व प्रदूषण टाळा", "नियमित तपासणी सुरू ठेवा", "संतुलित आहार घ्या", "नवीन लक्षण दिसल्यास डॉक्टरांचा सल्ला घ्या"],
    },

    "Pulmonary_Edema": {
        "en": ["Sit upright rather than lying flat to ease breathing", "Reduce salt intake in your meals", "Avoid strenuous physical activity", "Monitor for worsening breathlessness or swelling in legs", "Limit fluid intake as advised by your doctor", "Seek emergency care immediately if breathing becomes severely difficult"],
        "hi": ["सांस लेने में आसानी के लिए सीधे बैठें", "नमक की मात्रा कम करें", "अधिक शारीरिक परिश्रम से बचें", "सांस फूलने या सूजन बढ़ने पर ध्यान दें", "डॉक्टर की सलाह अनुसार पानी सीमित करें", "सांस लेना कठिन हो तो तुरंत सहायता लें"],
        "mr": ["श्वास सोपे होण्यासाठी सरळ बसा", "मिठाचे प्रमाण कमी करा", "जास्त शारीरिक श्रम टाळा", "त्रास वाढणे किंवा सूज येण्याकडे लक्ष द्या", "पाण्याचे सेवन मर्यादित करा", "त्रास झाल्यास त्वरित मदत घ्या"],
    },

    "Pulmonary_Fibrosis": {
        "en": ["Avoid smoking and exposure to dust or fumes", "Practice breathing exercises as recommended", "Stay physically active within your comfort level", "Get vaccinated against flu and pneumonia as advised", "Use supplemental oxygen if prescribed", "Seek emergency care if breathlessness worsens suddenly"],
        "hi": ["धूम्रपान और धूल-धुएं से बचें", "सुझाई गई सांस की कसरत करें", "सुविधा अनुसार सक्रिय रहें", "फ्लू और निमोनिया का टीका लगवाएं", "डॉक्टर की सलाह पर ऑक्सीजन उपयोग करें", "अचानक कठिनाई बढ़े तो सहायता लें"],
        "mr": ["धूम्रपान व धूळ-धूर टाळा", "शिफारस केलेले व्यायाम करा", "सुविधेनुसार सक्रिय राहा", "फ्लू व न्यूमोनियाची लस घ्या", "डॉक्टरांच्या सल्ल्याने ऑक्सिजन वापरा", "त्रास वाढल्यास मदत घ्या"],
    },

    "Tuberculosis": {
        "en": ["Complete the full course of prescribed medication", "Cover your mouth while coughing to avoid spreading infection", "Ensure good ventilation in your living space", "Eat a nutritious diet to support recovery", "Avoid alcohol and smoking during treatment", "Seek emergency care for coughing up blood or severe breathlessness"],
        "hi": ["दवाओं का पूरा कोर्स पूरा करें", "खांसते समय मुंह ढकें", "अच्छा वेंटिलेशन सुनिश्चित करें", "पौष्टिक आहार लें", "शराब और धूम्रपान से बचें", "खून आना/गंभीर सांस फूलने पर सहायता लें"],
        "mr": ["औषधोपचार पूर्ण करा", "खोकताना तोंड झाका", "चांगली हवा येण्याची व्यवस्था ठेवा", "पौष्टिक आहार घ्या", "मद्यपान व धूम्रपान टाळा", "रक्त येणे/तीव्र त्रास झाल्यास मदत घ्या"],
    },

}



# ===============================
# Prediction Function
# ===============================

# ===============================
# X-Ray Validation Function (Model 1)
# ===============================

def validate_xray(image_path):

    img = Image.open(image_path)

    img = img.convert("RGB")

    img = img.resize((224, 224))

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    prediction = validator_model.predict(img, verbose=0)

    confidence = prediction[0][0]

    print(f"[DEBUG] Validator confidence: {confidence:.4f}")

    if confidence > 0.5:
        return False

    return True


# ===============================
# X-Ray Validation Function (Model 2 - CLIP)
# Checks specifically whether the image is a
# CHEST x-ray, not just "any x-ray"
# ===============================

def validate_xray_clip(image_path):

    img = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(clip_device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(img, clip_text_tokens)
        probs = logits_per_image.softmax(dim=-1)[0]

    chest_xray_score = probs[0].item()   # index 0 = "a chest x-ray showing ribs and lungs"

    print(f"[DEBUG] CLIP chest x-ray score: {chest_xray_score:.4f}")

    return chest_xray_score > 0.5   # tune this after testing


def is_mostly_grayscale(image_path, threshold=15):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((100, 100))
    img_array = np.array(img)

    r = img_array[:, :, 0].astype(int)
    g = img_array[:, :, 1].astype(int)
    b = img_array[:, :, 2].astype(int)

    diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3
    avg_color_diff = diff.mean()

    print(f"[DEBUG] Avg color channel difference: {avg_color_diff:.2f}")

    return avg_color_diff < threshold

import cv2

def get_sharpness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 300))
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()

    print(f"[DEBUG] Sharpness score: {sharpness:.2f}")

    return sharpness


def predict_image(image_path):


    img = Image.open(
        image_path
    )


    img = img.convert(
        "RGB"
    )


    img = img.resize(
        (224,224)
    )


    img = img_to_array(
        img
    )


    img = np.expand_dims(
        img,
        axis=0
    )


    img = preprocess_input(
        img
    )



    prediction = model.predict(
        img
    )



    index = np.argmax(
        prediction
    )


    confidence = float(
        np.max(prediction) * 100
    )

    
    print(f"[DEBUG] Predicted: {class_names[index]} | Confidence: {confidence:.2f}%")

    top_3_idx = np.argsort(prediction[0])[-3:][::-1]
    for i in top_3_idx:
       print(f"    {class_names[i]}: {prediction[0][i]*100:.2f}%")


    disease = class_names[index]



    return disease, confidence




# =====================================================
# REAL GRAD-CAM HEATMAP
# =====================================================

def generate_gradcam(image_path, target_index):

    print("🔥 Starting Grad-CAM generation...")

    # -------------------------------------------------
    # Load original image
    # -------------------------------------------------

    original = Image.open(image_path).convert("RGB")

    original_np = np.array(original)

    # -------------------------------------------------
    # Prepare image exactly like prediction
    # -------------------------------------------------

    img = original.resize((224, 224))

    img_array = img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    # -------------------------------------------------
    # Find last convolutional layer automatically
    # -------------------------------------------------

    last_conv_layer = None

    for layer in reversed(model.layers):

        try:

            output_shape = layer.output.shape

            if len(output_shape) == 4:

                last_conv_layer = layer
                break

        except Exception:
            continue

    if last_conv_layer is None:

        raise Exception(
            "Could not find convolutional layer for Grad-CAM."
        )

    print(
        "🔥 Grad-CAM layer:",
        last_conv_layer.name
    )

    # -------------------------------------------------
    # Create Grad-CAM model
    # -------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # -------------------------------------------------
    # Calculate gradients
    # -------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        predicted_index = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[
            0,
            predicted_index
        ]

    gradients = tape.gradient(
        class_channel,
        conv_outputs
    )

    # -------------------------------------------------
    # Global average pooling
    # -------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

    # -------------------------------------------------
    # ReLU
    # -------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if max_value > 0:

        heatmap /= max_value

    heatmap = heatmap.numpy()

    # -------------------------------------------------
    # Resize heatmap to original image
    # -------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (
            original_np.shape[1],
            original_np.shape[0]
        )
    )

    # -------------------------------------------------
    # Convert heatmap to color
    # -------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # -------------------------------------------------
    # Create overlay
    # -------------------------------------------------

    overlay = cv2.addWeighted(
        original_np,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    # -------------------------------------------------
    # Save generated images
    # -------------------------------------------------

    heatmap_dir = os.path.join(
        "static",
        "heatmaps"
    )

    os.makedirs(
        heatmap_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    original_filename = (
        f"original_{timestamp}.jpg"
    )

    heatmap_filename = (
        f"heatmap_{timestamp}.jpg"
    )

    overlay_filename = (
        f"overlay_{timestamp}.jpg"
    )

    original_path = os.path.join(
        heatmap_dir,
        original_filename
    )

    heatmap_path = os.path.join(
        heatmap_dir,
        heatmap_filename
    )

    overlay_path = os.path.join(
        heatmap_dir,
        overlay_filename
    )

    # Save original
    cv2.imwrite(
        original_path,
        cv2.cvtColor(
            original_np,
            cv2.COLOR_RGB2BGR
        )
    )

    # Save heatmap
    cv2.imwrite(
        heatmap_path,
        cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_RGB2BGR
        )
    )

    # Save overlay
    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    print("✅ Grad-CAM generated successfully!")

    return {
        "original": "/" + original_path.replace("\\", "/"),
        "heatmap": "/" + heatmap_path.replace("\\", "/"),
        "overlay": "/" + overlay_path.replace("\\", "/")
    }



# ===============================
# Prediction API
# ===============================

@app.route("/api/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["image"]

    os.makedirs("uploads", exist_ok=True)

    path = os.path.join("uploads", file.filename)

    file.save(path)

    session["last_xray_path"] = path

    # ===============================
    # Validate Chest X-Ray
    # (Original validator AND CLIP must both agree)
    # ===============================

    is_chest_xray_v1 = validate_xray(path)
    is_chest_xray_v2 = validate_xray_clip(path)

    if not (is_chest_xray_v1 and is_chest_xray_v2):

        return jsonify({
            "error": "❌ Please upload a valid Chest X-Ray image."
        })

    is_mostly_grayscale(path)   # just running it to see the printed number, not blocking anything yet

    sharpness = get_sharpness(path)

    MAX_SHARPNESS = 1800   # real X-rays are soft (under ~800); screenshots are sharp (over ~4000)

    if sharpness > MAX_SHARPNESS:
        return jsonify({
            "error": "❌ This image looks like a screenshot or graphic, not an X-ray. Please upload a real chest X-Ray image."
        })

    # ===============================
    # Predict Disease
    # ===============================

    disease, confidence = predict_image(path)

    session["last_prediction"] = disease
    session["last_confidence"] = round(confidence, 2)
    session["last_class_index"] = class_names.index(disease)

    print("🔥 SAVED FOR HEATMAP:")
    print("Disease:", disease)
    print("Confidence:", round(confidence, 2))
    print("Class Index:", class_names.index(disease))

    # ===============================
    # Reject Low-Confidence Predictions
    # ===============================

    if confidence < MIN_CONFIDENCE:

        return jsonify({
            "error": "❌ This image doesn't clearly match a valid chest X-ray disease pattern we support. Please upload a clear chest X-ray."
        })

    if "user_id" in session:

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions
            (
                user_id,
                image_name,
                 predicted_disease,
                confidence
            )
            VALUES (?, ?, ?, ?)
        """,
        (
            session["user_id"],
            file.filename,
            disease,
            confidence
        ))

        conn.commit()
        conn.close()

    return jsonify({
        "prediction": disease,
        "confidence": round(confidence, 2)
    })

@app.route("/api/generate-report", methods=["POST"])
# def generate_report():
 
#     data = request.get_json()
 
#     disease = data.get("disease", "Unknown")
#     confidence = data.get("confidence", 0)
#     analysis_time = data.get("analysis_time", "--")
#     file_name = data.get("file_name", "chest_xray.png")
 
#     # Pull the actual uploaded X-ray saved during /api/predict
#     xray_path = session.get("last_xray_path")
#     print("[DEBUG] xray_path:", xray_path)
 
#     now = datetime.now()
#     report_id = f"LVAI-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
#     date_str = now.strftime("%d %B %Y, %I:%M %p")
 
#     disease_color = GREEN if disease == "Normal" else RED
 
#     os.makedirs("reports", exist_ok=True)
#     filename = f"LungVisionAI_Report_{disease.replace(' ', '_')}_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
#     filepath = os.path.join("reports", filename)
 
#     doc = SimpleDocTemplate(
#         filepath, pagesize=A4,
#         topMargin=0, bottomMargin=0, leftMargin=0, rightMargin=0
#     )
#     styles = getSampleStyleSheet()
#     story = []
 

#     title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, textColor=NAVY)
#     subtitle_style = ParagraphStyle("subtitle", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#3a4a63"))
 
#     header_table = Table(
#         [[Paragraph("LungVision AI", title_style)],
#          [Paragraph("AI-BASED CHEST X-RAY DISEASE PREDICTION REPORT", subtitle_style)]],
#         colWidths=[500]
#     )
#     header_table.setStyle(TableStyle([
#         ("LEFTPADDING", (0, 0), (-1, -1), 20),
#         ("TOPPADDING", (0, 0), (0, 0), 16),
#         ("BOTTOMPADDING", (0, 1), (0, 1), 14),
#         ("BACKGROUND", (0, 0), (-1, -1), colors.white),
#         ("LINEBELOW", (0, -1), (-1, -1), 4, NAVY),
#     ]))
#     story.append(header_table)
#     story.append(Spacer(1, 14))


 
#     # ---------- REPORT INFORMATION ----------
#     section_title_style = ParagraphStyle("sectionTitle", fontName="Helvetica-Bold", fontSize=12, textColor=NAVY)
#     # story.append(Paragraph("REPORT INFORMATION", section_title_style))
#     story.append(section_title_with_icon("report", "REPORT INFORMATION", section_title_style))
#     story.append(Spacer(1, 8))
 
#     label_style = ParagraphStyle("label", fontName="Helvetica", fontSize=9.5, textColor=GRAY_TEXT)
#     value_style = ParagraphStyle("value", fontName="Helvetica", fontSize=9.5, textColor=colors.black)
#     disease_value_style = ParagraphStyle("diseaseValue", fontName="Helvetica-Bold", fontSize=10.5, textColor=disease_color)
#     confidence_value_style = ParagraphStyle("confValue", fontName="Helvetica-Bold", fontSize=10.5, textColor=BLUE)
#     status_style = ParagraphStyle("status", fontName="Helvetica-Bold", fontSize=9.5, textColor=GREEN)
 
#     info_rows = [
#         [Paragraph("Report ID", label_style), Paragraph(report_id, value_style)],
#         [Paragraph("Date & Time", label_style), Paragraph(date_str, value_style)],
#         [Paragraph("Uploaded Image", label_style), Paragraph(file_name, value_style)],
#         [Paragraph("Predicted Disease", label_style), Paragraph(disease, disease_value_style)],
#         [Paragraph("Confidence Score", label_style), Paragraph(f"{confidence}%", confidence_value_style)],
#         [Paragraph("Analysis Status", label_style), Paragraph("Analysis Completed Successfully", status_style)],
#     ]
#     info_table = Table(info_rows, colWidths=[150, 200])
#     info_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8faff")),
#         ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
#         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#         ("LEFTPADDING", (0, 0), (-1, -1), 8),
#         ("TOPPADDING", (0, 0), (-1, -1), 6),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
#     ]))
 
    
#     xray_path = session.get("last_xray_path")

#     if xray_path and os.path.exists(xray_path):
#         try:
#             xray_img = RLImage(xray_path, width=170, height=170)
#         except Exception:
#             xray_img = Paragraph("", value_style)

#         row = Table([[info_table, xray_img]], colWidths=[360, 180])
#         row.setStyle(TableStyle([
#             ("VALIGN", (0, 0), (-1, -1), "TOP"),
#             ("LEFTPADDING", (1, 0), (1, 0), 12),
#         ]))
#         story.append(row)
#     else:
#         story.append(info_table)

 
#     # ---------- AI PREDICTION SUMMARY ----------
#     # story.append(Paragraph("AI PREDICTION SUMMARY", section_title_style))
#     story.append(section_title_with_icon("brain", "AI PREDICTION SUMMARY", section_title_style))
#     story.append(Spacer(1, 8))
 
#     summary_label_style = ParagraphStyle("sumLabel", fontName="Helvetica", fontSize=8, textColor=GRAY_TEXT, alignment=TA_CENTER)
#     summary_value_style = ParagraphStyle("sumValue", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE, alignment=TA_CENTER)
#     summary_disease_style = ParagraphStyle("sumDisease", fontName="Helvetica-Bold", fontSize=9.5, textColor=disease_color, alignment=TA_CENTER)
#     summary_green_style = ParagraphStyle("sumGreen", fontName="Helvetica-Bold", fontSize=9.5, textColor=GREEN, alignment=TA_CENTER)
 
#     summary_row = [
#         [Paragraph("Disease Detected", summary_label_style),
#          Paragraph("Confidence Level", summary_label_style),
#          Paragraph("Prediction Method", summary_label_style),
#          Paragraph("Image Validation", summary_label_style),
#          Paragraph("Processing Time", summary_label_style)],
#         [Paragraph(disease, summary_disease_style),
#          Paragraph(f"{confidence}%", summary_value_style),
#          Paragraph("Deep Learning CNN Model", summary_value_style),
#          Paragraph("Valid Chest X-Ray", summary_green_style),
#          Paragraph(str(analysis_time), summary_value_style)],
#     ]
#     summary_table = Table(summary_row, colWidths=[100, 100, 100, 100, 100])
#     summary_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f8ff")),
#         ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
#         ("TOPPADDING", (0, 0), (-1, -1), 10),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
#     ]))
#     story.append(summary_table)
#     story.append(Spacer(1, 18))
 
#     # ---------- DISCLAIMER + SYSTEM INFO ----------
#     disclaimer_title_style = ParagraphStyle("discTitle", fontName="Helvetica-Bold", fontSize=10, textColor=RED)
#     disclaimer_body_style = ParagraphStyle("discBody", fontName="Helvetica", fontSize=8.3, textColor=colors.HexColor("#5a3a3a"), leading=12)
 
#     disclaimer_content = [
#         # Paragraph("⚠ Medical Disclaimer", disclaimer_title_style),
#         section_title_with_icon("warning", "Medical Disclaimer", disclaimer_title_style, icon_size=14),
#         Spacer(1, 4),
#         Paragraph(
#             "This report contains an AI-generated prediction based on the uploaded chest X-ray image. "
#             "It is intended to assist healthcare professionals and should not be considered a confirmed "
#             "medical diagnosis.", disclaimer_body_style),
#         Spacer(1, 6),
#         Paragraph(
#             "The results should always be reviewed by a qualified radiologist or physician. Clinical "
#             "examination, laboratory tests, and additional imaging may be required before making any "
#             "medical decisions.", disclaimer_body_style),
#     ]
#     disclaimer_box = Table([[disclaimer_content]], colWidths=[260])
#     disclaimer_box.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
#         ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#f3c8c8")),
#         ("LEFTPADDING", (0, 0), (-1, -1), 12),
#         ("TOPPADDING", (0, 0), (-1, -1), 12),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
#     ]))
 
#     sys_label_style = ParagraphStyle("sysLabel", fontName="Helvetica-Bold", fontSize=8.5)
#     sys_value_style = ParagraphStyle("sysValue", fontName="Helvetica", fontSize=8.5)
#     sys_rows = [
#         [Paragraph("System Name", sys_label_style), Paragraph("LungVision AI", sys_value_style)],
#         [Paragraph("AI Model", sys_label_style), Paragraph("Convolutional Neural Network (CNN)", sys_value_style)],
#         [Paragraph("Image Type", sys_label_style), Paragraph("Chest X-Ray", sys_value_style)],
#         [Paragraph("Supported Diseases", sys_label_style), Paragraph(", ".join(class_names), sys_value_style)],
#         [Paragraph("Generated By", sys_label_style), Paragraph("LungVision AI Prediction System", sys_value_style)],
#         [Paragraph("Report Version", sys_label_style), Paragraph("1.0", sys_value_style)],
#     ]
#     sys_table = Table(sys_rows, colWidths=[100, 160])
#     sys_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8faff")),
#         ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
#         ("TOPPADDING", (0, 0), (-1, -1), 5),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
#     ]))
 
#     two_col = Table([[disclaimer_box, sys_table]], colWidths=[270, 270])
#     two_col.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (1, 0), (1, 0), 14)]))
#     story.append(two_col)
#     story.append(Spacer(1, 18))
 
#     # ---------- AUTHENTICATION STRIP ----------
#     auth_label_style = ParagraphStyle("authLabel", fontName="Helvetica", fontSize=8, alignment=TA_CENTER)
#     auth_value_style = ParagraphStyle("authValue", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN, alignment=TA_CENTER)
#     auth_value_black_style = ParagraphStyle("authValueBlack", fontName="Helvetica-Bold", fontSize=9, alignment=TA_CENTER)
 
#     auth_row = [[
#         Paragraph("Report Status<br/><font color='#1a9c5a'>AI Analysis Completed</font>", auth_label_style),
#         Paragraph(f"Generated On<br/><b>{date_str}</b>", auth_label_style),
#         Paragraph("Report Version<br/><b>1.0</b>", auth_label_style),
#     ]]
#     auth_table = Table(auth_row, colWidths=[180, 180, 180])
#     auth_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4fbf6")),
#         ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cdeedc")),
#         ("TOPPADDING", (0, 0), (-1, -1), 10),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
#     ]))
#     story.append(auth_table)
#     story.append(Spacer(1, 14))


#  # ===============================
#     # Footer
#     # ===============================
#     footer_style = ParagraphStyle("footer", fontName="Helvetica", fontSize=8.5, textColor=colors.white)
#     footer_bold_style = ParagraphStyle("footerBold", fontName="Helvetica-Bold", fontSize=10, textColor=colors.white)
#     footer_row = [[
#         Paragraph("<b>LungVision AI</b><br/>AI-Powered Chest X-Ray Disease Detection System", footer_style),
#         Paragraph("Your Health, Our Mission", footer_style),
#     ]]
#     footer_table = Table(footer_row, colWidths=[300, 200])
#     footer_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), NAVY),
#         ("LEFTPADDING", (0, 0), (0, 0), 20),
#         ("TOPPADDING", (0, 0), (-1, -1), 12),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
#         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#     ]))
#     story.append(footer_table)
 
#     doc.build(story)
 
#     return send_file(filepath, as_attachment=True, download_name=filename)

#  # ===============================
#     # Build the PDF and send it back
#     # ===============================

#     doc.build(story)

#     return send_file(
#         filepath,
#         as_attachment=True,
#         download_name=filename,
#         mimetype="application/pdf"
#     )
@app.route("/api/generate-report", methods=["POST"])
def generate_report():

    data = request.get_json()

    disease = data.get("disease", "Unknown")
    confidence = data.get("confidence", 0)
    analysis_time = data.get("analysis_time", "--")
    file_name = data.get("file_name", "chest_xray.png")

    xray_path = session.get("last_xray_path")
    print("[DEBUG] xray_path:", xray_path)

    now = datetime.now()
    report_id = f"LVAI-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    date_str = now.strftime("%d %B %Y, %I:%M %p")

    disease_color = GREEN if disease == "Normal" else RED

    os.makedirs("reports", exist_ok=True)
    filename = f"LungVisionAI_Report_{disease.replace(' ', '_')}_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("reports", filename)

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        topMargin=0, bottomMargin=0, leftMargin=0, rightMargin=0
    )
    styles = getSampleStyleSheet()
    story = []

    # ---------- HEADER (fixed — no overlap) ----------
    title_style = ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=22, textColor=NAVY,
        leading=26, spaceAfter=0
    )
    subtitle_style = ParagraphStyle(
        "subtitle", fontName="Helvetica-Bold", fontSize=9,
        textColor=colors.HexColor("#3a4a63"), leading=12, spaceBefore=2
    )
    brand_icon_style = ParagraphStyle(
        "brandIcon", fontName="Helvetica-Bold", fontSize=16,
        textColor=BLUE, alignment=TA_CENTER, leading=18
    )
    brand_icon_cell = Table([[Paragraph("L", brand_icon_style)]], colWidths=[36], rowHeights=[36])
    brand_icon_cell.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.4, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    title_block = Table(
        [[Paragraph("LungVision AI", title_style)],
         [Paragraph("AI-BASED CHEST X-RAY DISEASE PREDICTION REPORT", subtitle_style)]],
        colWidths=[460],
        rowHeights=[30, 16]
    )
    title_block.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 0),
        ("VALIGN", (0, 0), (0, 0), "BOTTOM"),
        ("VALIGN", (0, 1), (0, 1), "TOP"),
    ]))

    header_row = Table([[brand_icon_cell, title_block]], colWidths=[56, 460])
    header_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 20),
        ("LEFTPADDING", (1, 0), (1, 0), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))

    story.append(header_row)

    rule = Table([[""]], colWidths=[555])
    rule.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 4, NAVY)]))
    story.append(rule)

    story.append(Spacer(1, 14))

    # ---------- REPORT INFORMATION ----------
    section_title_style = ParagraphStyle("sectionTitle", fontName="Helvetica-Bold", fontSize=12, textColor=NAVY)
    story.append(section_title_with_icon("report", "REPORT INFORMATION", section_title_style))
    story.append(Spacer(1, 8))

    label_style = ParagraphStyle("label", fontName="Helvetica", fontSize=9.5, textColor=GRAY_TEXT)
    value_style = ParagraphStyle("value", fontName="Helvetica", fontSize=9.5, textColor=colors.black)
    disease_value_style = ParagraphStyle("diseaseValue", fontName="Helvetica-Bold", fontSize=10.5, textColor=disease_color)
    confidence_value_style = ParagraphStyle("confValue", fontName="Helvetica-Bold", fontSize=10.5, textColor=BLUE)
    status_style = ParagraphStyle("status", fontName="Helvetica-Bold", fontSize=9.5, textColor=GREEN)

    info_rows = [
        [Paragraph("Report ID", label_style), Paragraph(report_id, value_style)],
        [Paragraph("Date & Time", label_style), Paragraph(date_str, value_style)],
        [Paragraph("Uploaded Image", label_style), Paragraph(file_name, value_style)],
        [Paragraph("Predicted Disease", label_style), Paragraph(disease, disease_value_style)],
        [Paragraph("Confidence Score", label_style), Paragraph(f"{confidence}%", confidence_value_style)],
        [Paragraph("Analysis Status", label_style), Paragraph("Analysis Completed Successfully", status_style)],
    ]
    info_table = Table(info_rows, colWidths=[150, 200])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8faff")),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    if xray_path and os.path.exists(xray_path):
        try:
            xray_img = RLImage(xray_path, width=170, height=170)
            row = Table([[info_table, xray_img]], colWidths=[360, 180])
            row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (1, 0), (1, 0), 12),
            ]))
            story.append(row)
        except Exception as e:
            print("[DEBUG] Failed to load xray image:", e)
            story.append(info_table)
    else:
        print("[DEBUG] xray_path missing or file not found on disk:", xray_path)
        story.append(info_table)

    story.append(Spacer(1, 18))

    # ---------- AI PREDICTION SUMMARY ----------
    story.append(section_title_with_icon("brain", "AI PREDICTION SUMMARY", section_title_style))
    story.append(Spacer(1, 8))

    summary_label_style = ParagraphStyle("sumLabel", fontName="Helvetica", fontSize=8, textColor=GRAY_TEXT, alignment=TA_CENTER)
    summary_value_style = ParagraphStyle("sumValue", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE, alignment=TA_CENTER)
    summary_disease_style = ParagraphStyle("sumDisease", fontName="Helvetica-Bold", fontSize=9.5, textColor=disease_color, alignment=TA_CENTER)
    summary_green_style = ParagraphStyle("sumGreen", fontName="Helvetica-Bold", fontSize=9.5, textColor=GREEN, alignment=TA_CENTER)

    summary_row = [
        [Paragraph("Disease Detected", summary_label_style),
         Paragraph("Confidence Level", summary_label_style),
         Paragraph("Prediction Method", summary_label_style),
         Paragraph("Image Validation", summary_label_style),
         Paragraph("Processing Time", summary_label_style)],
        [Paragraph(disease, summary_disease_style),
         Paragraph(f"{confidence}%", summary_value_style),
         Paragraph("Deep Learning CNN Model", summary_value_style),
         Paragraph("Valid Chest X-Ray", summary_green_style),
         Paragraph(str(analysis_time), summary_value_style)],
    ]
    summary_table = Table(summary_row, colWidths=[100, 100, 100, 100, 100])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f8ff")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 18))

    # ---------- DISCLAIMER + SYSTEM INFO ----------
    disclaimer_title_style = ParagraphStyle("discTitle", fontName="Helvetica-Bold", fontSize=10, textColor=RED)
    disclaimer_body_style = ParagraphStyle("discBody", fontName="Helvetica", fontSize=8.3, textColor=colors.HexColor("#5a3a3a"), leading=12)

    disclaimer_content = [
        section_title_with_icon("warning", "Medical Disclaimer", disclaimer_title_style, icon_size=14),
        Spacer(1, 4),
        Paragraph(
            "This report contains an AI-generated prediction based on the uploaded chest X-ray image. "
            "It is intended to assist healthcare professionals and should not be considered a confirmed "
            "medical diagnosis.", disclaimer_body_style),
        Spacer(1, 6),
        Paragraph(
            "The results should always be reviewed by a qualified radiologist or physician. Clinical "
            "examination, laboratory tests, and additional imaging may be required before making any "
            "medical decisions.", disclaimer_body_style),
    ]
    disclaimer_box = Table([[disclaimer_content]], colWidths=[260])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#f3c8c8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    sys_label_style = ParagraphStyle("sysLabel", fontName="Helvetica-Bold", fontSize=8.5)
    sys_value_style = ParagraphStyle("sysValue", fontName="Helvetica", fontSize=8.5)
    sys_rows = [
        [Paragraph("System Name", sys_label_style), Paragraph("LungVision AI", sys_value_style)],
        [Paragraph("AI Model", sys_label_style), Paragraph("Convolutional Neural Network (CNN)", sys_value_style)],
        [Paragraph("Image Type", sys_label_style), Paragraph("Chest X-Ray", sys_value_style)],
        [Paragraph("Supported Diseases", sys_label_style), Paragraph(", ".join(class_names), sys_value_style)],
        [Paragraph("Generated By", sys_label_style), Paragraph("LungVision AI Prediction System", sys_value_style)],
        [Paragraph("Report Version", sys_label_style), Paragraph("1.0", sys_value_style)],
    ]
    sys_table = Table(sys_rows, colWidths=[100, 160])
    sys_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8faff")),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    two_col = Table([[disclaimer_box, sys_table]], colWidths=[270, 270])
    two_col.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (1, 0), (1, 0), 14)]))
    story.append(two_col)
    story.append(Spacer(1, 18))

    # ---------- AUTHENTICATION STRIP ----------
    auth_label_style = ParagraphStyle("authLabel", fontName="Helvetica", fontSize=8, alignment=TA_CENTER)

    auth_row = [[
        Paragraph("Report Status<br/><font color='#1a9c5a'>AI Analysis Completed</font>", auth_label_style),
        Paragraph(f"Generated On<br/><b>{date_str}</b>", auth_label_style),
        Paragraph("Report Version<br/><b>1.0</b>", auth_label_style),
    ]]
    auth_table = Table(auth_row, colWidths=[180, 180, 180])
    auth_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4fbf6")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cdeedc")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(auth_table)
    story.append(Spacer(1, 14))

    # ---------- FOOTER ----------
    footer_style = ParagraphStyle("footer", fontName="Helvetica", fontSize=8.5, textColor=colors.white)
    footer_row = [[
        Paragraph("<b>LungVision AI</b><br/>AI-Powered Chest X-Ray Disease Detection System", footer_style),
        Paragraph("Your Health, Our Mission", footer_style),
    ]]
    footer_table = Table(footer_row, colWidths=[300, 200])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (0, 0), 20),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(footer_table)

    # ---------- Build the PDF and send it back ----------
    doc.build(story)

    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf"
    )

# @app.route("/api/generate-report", methods=["POST"])
# def generate_report():
 
#     data = request.get_json()
 
#     disease = data.get("disease", "Unknown")
#     confidence = data.get("confidence", 0)
#     analysis_time = data.get("analysis_time", "--")
#     file_name = data.get("file_name", "chest_xray.png")
 
#     # Pull the actual uploaded X-ray saved during /api/predict
#     xray_path = session.get("last_xray_path")
 
#     now = datetime.now()
#     report_id = f"LVAI-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
#     date_str = now.strftime("%d %B %Y, %I:%M %p")
 
#     disease_color = GREEN if disease == "Normal" else RED
 
#     os.makedirs("reports", exist_ok=True)
#     filename = f"LungVisionAI_Report_{disease.replace(' ', '_')}_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
#     filepath = os.path.join("reports", filename)
 
#     doc = SimpleDocTemplate(
#         filepath, pagesize=A4,
#         topMargin=0, bottomMargin=0, leftMargin=0, rightMargin=0
#     )
#     styles = getSampleStyleSheet()
#     story = []
 

#     title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, textColor=NAVY)
#     subtitle_style = ParagraphStyle("subtitle", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#3a4a63"))
 
#     header_table = Table(
#         [[Paragraph("LungVision AI", title_style)],
#          [Paragraph("AI-BASED CHEST X-RAY DISEASE PREDICTION REPORT", subtitle_style)]],
#         colWidths=[500]
#     )
#     header_table.setStyle(TableStyle([
#         ("LEFTPADDING", (0, 0), (-1, -1), 20),
#         ("TOPPADDING", (0, 0), (0, 0), 16),
#         ("BOTTOMPADDING", (0, 1), (0, 1), 14),
#         ("BACKGROUND", (0, 0), (-1, -1), colors.white),
#         ("LINEBELOW", (0, -1), (-1, -1), 4, NAVY),
#     ]))
#     story.append(header_table)
#     story.append(Spacer(1, 14))
 
#     # ---------- REPORT INFORMATION ----------
#     section_title_style = ParagraphStyle("sectionTitle", fontName="Helvetica-Bold", fontSize=12, textColor=NAVY)
#     # story.append(Paragraph("REPORT INFORMATION", section_title_style))
#     story.append(section_title_with_icon("report", "REPORT INFORMATION", section_title_style))
#     story.append(Spacer(1, 8))
 
#     label_style = ParagraphStyle("label", fontName="Helvetica", fontSize=9.5, textColor=GRAY_TEXT)
#     value_style = ParagraphStyle("value", fontName="Helvetica", fontSize=9.5, textColor=colors.black)
#     disease_value_style = ParagraphStyle("diseaseValue", fontName="Helvetica-Bold", fontSize=10.5, textColor=disease_color)
#     confidence_value_style = ParagraphStyle("confValue", fontName="Helvetica-Bold", fontSize=10.5, textColor=BLUE)
#     status_style = ParagraphStyle("status", fontName="Helvetica-Bold", fontSize=9.5, textColor=GREEN)
 
#     info_rows = [
#         [Paragraph("Report ID", label_style), Paragraph(report_id, value_style)],
#         [Paragraph("Date & Time", label_style), Paragraph(date_str, value_style)],
#         [Paragraph("Uploaded Image", label_style), Paragraph(file_name, value_style)],
#         [Paragraph("Predicted Disease", label_style), Paragraph(disease, disease_value_style)],
#         [Paragraph("Confidence Score", label_style), Paragraph(f"{confidence}%", confidence_value_style)],
#         [Paragraph("Analysis Status", label_style), Paragraph("Analysis Completed Successfully", status_style)],
#     ]
#     info_table = Table(info_rows, colWidths=[150, 200])
#     info_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8faff")),
#         ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
#         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#         ("LEFTPADDING", (0, 0), (-1, -1), 8),
#         ("TOPPADDING", (0, 0), (-1, -1), 6),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
#     ]))
 
    
#     xray_path = session.get("last_xray_path")

#     if xray_path and os.path.exists(xray_path):
#         try:
#             xray_img = RLImage(xray_path, width=170, height=170)
#         except Exception:
#             xray_img = Paragraph("", value_style)

#         row = Table([[info_table, xray_img]], colWidths=[360, 180])
#         row.setStyle(TableStyle([
#             ("VALIGN", (0, 0), (-1, -1), "TOP"),
#             ("LEFTPADDING", (1, 0), (1, 0), 12),
#         ]))
#         story.append(row)
#     else:
#         story.append(info_table)

 
#     # ---------- AI PREDICTION SUMMARY ----------
#     # story.append(Paragraph("AI PREDICTION SUMMARY", section_title_style))
#     story.append(section_title_with_icon("brain", "AI PREDICTION SUMMARY", section_title_style))
#     story.append(Spacer(1, 8))
 
#     summary_label_style = ParagraphStyle("sumLabel", fontName="Helvetica", fontSize=8, textColor=GRAY_TEXT, alignment=TA_CENTER)
#     summary_value_style = ParagraphStyle("sumValue", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE, alignment=TA_CENTER)
#     summary_disease_style = ParagraphStyle("sumDisease", fontName="Helvetica-Bold", fontSize=9.5, textColor=disease_color, alignment=TA_CENTER)
#     summary_green_style = ParagraphStyle("sumGreen", fontName="Helvetica-Bold", fontSize=9.5, textColor=GREEN, alignment=TA_CENTER)
 
#     summary_row = [
#         [Paragraph("Disease Detected", summary_label_style),
#          Paragraph("Confidence Level", summary_label_style),
#          Paragraph("Prediction Method", summary_label_style),
#          Paragraph("Image Validation", summary_label_style),
#          Paragraph("Processing Time", summary_label_style)],
#         [Paragraph(disease, summary_disease_style),
#          Paragraph(f"{confidence}%", summary_value_style),
#          Paragraph("Deep Learning CNN Model", summary_value_style),
#          Paragraph("Valid Chest X-Ray", summary_green_style),
#          Paragraph(str(analysis_time), summary_value_style)],
#     ]
#     summary_table = Table(summary_row, colWidths=[100, 100, 100, 100, 100])
#     summary_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f8ff")),
#         ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
#         ("TOPPADDING", (0, 0), (-1, -1), 10),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
#     ]))
#     story.append(summary_table)
#     story.append(Spacer(1, 18))
 
#     # ---------- DISCLAIMER + SYSTEM INFO ----------
#     disclaimer_title_style = ParagraphStyle("discTitle", fontName="Helvetica-Bold", fontSize=10, textColor=RED)
#     disclaimer_body_style = ParagraphStyle("discBody", fontName="Helvetica", fontSize=8.3, textColor=colors.HexColor("#5a3a3a"), leading=12)
 
#     disclaimer_content = [
#         # Paragraph("⚠ Medical Disclaimer", disclaimer_title_style),
#         section_title_with_icon("warning", "Medical Disclaimer", disclaimer_title_style, icon_size=14),
#         Spacer(1, 4),
#         Paragraph(
#             "This report contains an AI-generated prediction based on the uploaded chest X-ray image. "
#             "It is intended to assist healthcare professionals and should not be considered a confirmed "
#             "medical diagnosis.", disclaimer_body_style),
#         Spacer(1, 6),
#         Paragraph(
#             "The results should always be reviewed by a qualified radiologist or physician. Clinical "
#             "examination, laboratory tests, and additional imaging may be required before making any "
#             "medical decisions.", disclaimer_body_style),
#     ]
#     disclaimer_box = Table([[disclaimer_content]], colWidths=[260])
#     disclaimer_box.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
#         ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#f3c8c8")),
#         ("LEFTPADDING", (0, 0), (-1, -1), 12),
#         ("TOPPADDING", (0, 0), (-1, -1), 12),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
#     ]))
 
#     sys_label_style = ParagraphStyle("sysLabel", fontName="Helvetica-Bold", fontSize=8.5)
#     sys_value_style = ParagraphStyle("sysValue", fontName="Helvetica", fontSize=8.5)
#     sys_rows = [
#         [Paragraph("System Name", sys_label_style), Paragraph("LungVision AI", sys_value_style)],
#         [Paragraph("AI Model", sys_label_style), Paragraph("Convolutional Neural Network (CNN)", sys_value_style)],
#         [Paragraph("Image Type", sys_label_style), Paragraph("Chest X-Ray", sys_value_style)],
#         [Paragraph("Supported Diseases", sys_label_style), Paragraph(", ".join(class_names), sys_value_style)],
#         [Paragraph("Generated By", sys_label_style), Paragraph("LungVision AI Prediction System", sys_value_style)],
#         [Paragraph("Report Version", sys_label_style), Paragraph("1.0", sys_value_style)],
#     ]
#     sys_table = Table(sys_rows, colWidths=[100, 160])
#     sys_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8faff")),
#         ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
#         ("TOPPADDING", (0, 0), (-1, -1), 5),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
#     ]))
 
#     two_col = Table([[disclaimer_box, sys_table]], colWidths=[270, 270])
#     two_col.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (1, 0), (1, 0), 14)]))
#     story.append(two_col)
#     story.append(Spacer(1, 18))
 
#     # ---------- AUTHENTICATION STRIP ----------
#     auth_label_style = ParagraphStyle("authLabel", fontName="Helvetica", fontSize=8, alignment=TA_CENTER)
#     auth_value_style = ParagraphStyle("authValue", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN, alignment=TA_CENTER)
#     auth_value_black_style = ParagraphStyle("authValueBlack", fontName="Helvetica-Bold", fontSize=9, alignment=TA_CENTER)
 
#     auth_row = [[
#         Paragraph("Report Status<br/><font color='#1a9c5a'>AI Analysis Completed</font>", auth_label_style),
#         Paragraph(f"Generated On<br/><b>{date_str}</b>", auth_label_style),
#         Paragraph("Report Version<br/><b>1.0</b>", auth_label_style),
#     ]]
#     auth_table = Table(auth_row, colWidths=[180, 180, 180])
#     auth_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4fbf6")),
#         ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cdeedc")),
#         ("TOPPADDING", (0, 0), (-1, -1), 10),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
#     ]))
#     story.append(auth_table)
#     story.append(Spacer(1, 14))
 
#     # ---------- FOOTER ----------
#     footer_style = ParagraphStyle("footer", fontName="Helvetica", fontSize=8.5, textColor=colors.white)
#     footer_bold_style = ParagraphStyle("footerBold", fontName="Helvetica-Bold", fontSize=10, textColor=colors.white)
#     footer_row = [[
#         Paragraph("<b>LungVision AI</b><br/>AI-Powered Chest X-Ray Disease Detection System", footer_style),
#         Paragraph("Your Health, Our Mission", footer_style),
#     ]]
#     footer_table = Table(footer_row, colWidths=[300, 200])
#     footer_table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, -1), NAVY),
#         ("LEFTPADDING", (0, 0), (0, 0), 20),
#         ("TOPPADDING", (0, 0), (-1, -1), 12),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
#         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#     ]))
#     story.append(footer_table)
 
#     doc.build(story)
 
#     return send_file(filepath, as_attachment=True, download_name=filename)





def section_title_with_icon(icon_name, title_text, style, icon_size=16):
    """
    Returns a styled title Paragraph for a report section.
    (Simplified — shows text only, no separate icon image needed.)
    """
    return Paragraph(title_text, style)

@app.route("/api/text-to-speech", methods=["POST"])
def text_to_speech():

    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        tts = gTTS(text=text, lang=lang)

        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return audio_buffer.read(), 200, {
            "Content-Type": "audio/mpeg"
        }

    except Exception as e:
        print("[DEBUG] gTTS Error:", e)
        return jsonify({"error": "Could not generate speech"}), 500



# ===============================
# Home Care Precautions API (NEW)
# ===============================

@app.route("/api/precautions/<disease>", methods=["GET"])
def get_precautions(disease):

    disease_data = PRECAUTIONS.get(disease)

    if not disease_data:
        return jsonify({
            "error": f"No precautions found for '{disease}'"
        }), 404

    return jsonify({
        "disease": disease,
        "precautions": disease_data
    })




# ===============================
# Chatbot API
# ===============================


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()


    user_message = data.get(
        "message",
        ""
    )


    latitude = data.get(
        "latitude"
    )


    longitude = data.get(
        "longitude"
    )


    print("Message:", user_message)

    print("Latitude:", latitude)

    print("Longitude:", longitude)



    # ===============================
    # Hospital Search
    # ===============================


    if is_hospital_query(
        user_message
    ):


        if latitude is None or longitude is None:


            return jsonify({

                "reply":
                "📍 Please allow location access to find nearby hospitals."

            })



        hospitals = get_nearby_hospitals(

            latitude,

            longitude

        )



        if not hospitals:


            return jsonify({

                "reply":
                "❌ No hospitals found within 10 km."

            })



        reply = "## 🏥 Nearby Hospitals\n\n"



        for i, hospital in enumerate(
            hospitals,
            start=1
        ):


            reply += (

                f"### {i}. {hospital['name']}\n"

                f"- 📍 Distance: {hospital['distance']} km\n"

                f"- 🗺️ [Open Google Maps]({hospital['maps']})\n\n"

            )



        return jsonify({

            "reply": reply

        })





    # ===============================
    # Gemini AI
    # ===============================


    try:


        response = ask_gemini(

            user_message

        )


        return jsonify({

            "reply": response

        })



    except Exception as e:


        print(
            "Chat Error:",
            
        )


        return jsonify({

            "reply":
            "⚠️ Unable to connect to AI assistant."

        })

# ===============================
# Run Flask Server
# ===============================

@app.route('/coupon')
def coupon():

    patient_name = None
    patient_email = None

    if "user_id" in session:

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT full_name, email FROM users WHERE id=?",
            (session["user_id"],)
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            patient_name = user["full_name"]
            patient_email = user["email"]

    return render_template(
        "coupon.html",
        patient_name=patient_name,
        patient_email=patient_email
    )

@app.route("/verify-coupon")
def verify_coupon():
    coupon_id = request.args.get("couponId")

    # TODO: replace with real DB lookup
    # e.g. check coupon table for existence, expiry, used flag
    status = "valid"   # "valid" | "invalid" | "expired"

    return render_template(
        "verify_coupon.html",
        coupon_id=coupon_id,
        status=status,
        patient_name=None,   # pull from DB if available
        disease=None
    )





@app.route("/api/verify-password", methods=["POST"])
def verify_password():

    if "user_id" not in session:
        return jsonify({"success": False, "message": "Not logged in."}), 401

    data = request.get_json()
    password = data.get("password", "")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE id=?",
        (session["user_id"],)       
    )

    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return jsonify({"success": True})

    return jsonify({"success": False, "message": "Incorrect password."}), 









# ===============================
# Run Flask Server
# ===============================


if __name__ == "__main__":


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    ) 