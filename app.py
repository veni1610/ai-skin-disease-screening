print("INFERENCE FILE LOADED")
import streamlit as st
from app.inference import predict
import tempfile
import os

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

    #st.subheader("Explainable AI (Model Attention Map)")
    #st.image(result["heatmap"])

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if st.button("🔍 Analyze"):
        st.write("Analyze button pressed")
        st.markdown("---")
        st.caption("⚠ This AI-based screening tool provides preliminary assessment only and should not be considered a medical diagnosis. Please consult a certified dermatologist for professional evaluation.")

        #st.write("Calling predict function")
        result = predict(temp_path)

        #st.write("Prediction kazhinj")
        #st.write(result)

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

            st.subheader("Reference Images for Comparison")

            image_folder = f"assets/{disease}"

            if os.path.exists(image_folder):
                images = os.listdir(image_folder)

                cols = st.columns(3)

                for i, img in enumerate(images[:2]):  # show up to 3 images
                    with cols[i]:
                        st.image(
                            os.path.join(image_folder, img),
                            caption=f"Example of {disease}",
                            use_container_width=True
                        )

            #st.subheader("Expainable AI (Model Attention Map)")
            #st.image(result["heatmap"], caption="Highlighted regions influencing the predction")
            if "heatmap" in result:
                st.subheader("Explainable AI (Model Attention Map)")
                st.image(result["heatmap"], caption="Highlighted regions influencing the prediction")

                st.caption(
    "This heatmap shows the regions of the skin image that the AI model focused on when making its prediction. "
    "Red and yellow areas indicate higher importance, meaning these regions contributed more to the decision. "
    "Blue or darker areas had little influence on the prediction."
                )
            
        if risk == "High":
            st.error("Recommendation: The detected condition may be serious. Please consult a certified dermatologist immediately for further examination and possible biopsy.")
        elif risk == "Moderate":
            st.warning("Recommendation: It is advisable to seek medical consultation. Early evaluation can prevent worsening of symptoms.")
        else:
            st.success("Recommendation: This appears to be a low-risk condition. Maintain proper skin hygiene and monitor for any changes.")