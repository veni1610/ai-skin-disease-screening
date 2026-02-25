import streamlit as st
from app.inference import predict
import tempfile


st.set_page_config(page_title="AI Skin Screening", layout="centered")


st.title("🩺 AI Skin Disease Screening System")
st.markdown("Upload an image and answer a few questions.")

uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

# --- Symptom Questions ---
st.subheader("Symptom Check")

itching = st.radio("Is the area itchy?", ["Yes", "No"])
pain = st.radio("Is there pain?", ["Yes", "No"])
bleeding = st.radio("Is there bleeding?", ["Yes", "No"])
duration = st.selectbox(
    "How long has it been present?",
    ["Less than 1 week", "1-4 weeks", "More than 1 month"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if st.button("🔍 Analyze"):
        st.markdown("---")
        st.caption("⚠ This AI-based screening tool provides preliminary assessment only and should not be considered a medical diagnosis. Please consult a certified dermatologist for professional evaluation.")

        result = predict(temp_path)

        disease = result["disease"]
        confidence = result["confidence"]
        risk = result["risk"]
        explanation = result["explanation"]

        # --- Risk Adjustment Based on Symptoms ---
        base_risk = result["risk"]
        disease = result["disease"]
        final_risk = base_risk

        dangerous_conditions = ["melanoma", "bcc"]

        if disease in dangerous_conditions:
            if bleeding == "Yes" or duration == "More than 1 month":
                final_risk = "High"
        elif base_risk == "Low":
            if bleeding == "Yes":
                final_risk = "Moderate"
        
        consultation = "Not Urgent"

        if bleeding == "Yes":
            consultation = "Consult Dermatologist"

        elif pain == "Yes" and duration == "More than 1 month":
            consultation = "Consult Dermatologist"

        elif itching == "Yes" and duration == "More than 1 month":
            consultation = "Recommended Consultation"

        with st.container():
            st.success(f"Predicted Disease: {disease}")
            st.info(f"AI Certanity Level: {confidence}")
            st.warning(f"Medical Risk Level: {risk}")
            st.write(f"Risk Explanation: {result['risk_explanation']}")
            
            st.subheader("Consultation Advice")
            st.write(consultation)
            
            st.subheader("Condition Explanation")
            st.write(explanation)   
            st.write(f"Condition Explanation: {explanation}")
            
        if risk == "High":
            st.error("Recommendation: The detected condition may be serious. Please consult a certified dermatologist immediately for further examination and possible biopsy.")
        elif risk == "Moderate":
            st.warning("Recommendation: It is advisable to seek medical consultation. Early evaluation can prevent worsening of symptoms.")
        else:
            st.success("Recommendation: This appears to be a low-risk condition. Maintain proper skin hygiene and monitor for any changes.").success("Recommendation: Monitor condition and maintain hygiene.")