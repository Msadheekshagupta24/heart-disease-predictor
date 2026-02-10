from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import os
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load("heart_model.pkl")

# Store last prediction globally (for PDF download)
last_result = {}
pdf_path = "heart_report.pdf"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global last_result

    values = [float(x) for x in request.form.values()]
    final_values = np.array(values).reshape(1, -1)

    prediction = model.predict(final_values)[0]
    prob = model.predict_proba(final_values)[0][1]
    prob_percent = round(prob * 100, 2)

    # Risk text
    if prediction == 1:
        result = "⚠️ High Risk of Heart Disease"
        tips = [
            "Consult a cardiologist immediately.",
            "Reduce cholesterol and salt intake.",
            "Exercise regularly under medical guidance.",
            "Avoid smoking and alcohol.",
            "Monitor blood pressure and sugar levels."
        ]
    else:
        result = "✅ Low Risk (Healthy)"
        tips = [
            "Maintain a balanced diet.",
            "Exercise at least 30 minutes daily.",
            "Do regular health checkups.",
            "Manage stress and sleep well.",
            "Avoid smoking and junk food."
        ]

    # Save for PDF
    last_result = {
        "prob": prob_percent,
        "result": result,
        "tips": tips,
        "time": datetime.now().strftime("%d %B %Y, %H:%M")
    }

    return render_template(
        "index.html",
        prediction_text=result,
        probability=prob_percent,
        tips=tips
    )


@app.route("/download")
def download():
    # Create PDF
    doc = SimpleDocTemplate(pdf_path)

    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = "HYSMyeongJo-Medium"

    elements = []

    elements.append(Paragraph("Heart Disease Prediction Report", style))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Result: {last_result.get('result','')}", style))
    elements.append(Paragraph(f"Probability: {last_result.get('prob','')}%", style))
    elements.append(Paragraph(f"Date & Time: {last_result.get('time','')}", style))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Health Recommendations:", style))
    elements.append(Spacer(1, 10))

    for tip in last_result.get("tips", []):
        elements.append(Paragraph(f"• {tip}", style))
        elements.append(Spacer(1, 8))

    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
     app.run(host="0.0.0.0", port=10000)
