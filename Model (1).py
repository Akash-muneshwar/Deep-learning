import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from openai import OpenAI
from fpdf import FPDF
import base64

# Load the trained model
model = load_model("/content/disease_prediction_ann_model (2).h5")

# Load the label classes
label_classes = np.load("/content/label_classes (2).npy", allow_pickle=True)

# Load the scaler
scaler = joblib.load("/content/scaler (1).pkl")

# Load the symptom names
with open("/content/symptom_order.txt", "r") as file:
    symptom_names = file.read().splitlines()

# Initialize OpenAI client
client = OpenAI(api_key="Replace with your actual OpenAI API key")  # Replace with your actual OpenAI API key

# Function to generate chatbot response
def generate_chatbot_response(predicted_disease):
    input_message = f"The disease is {predicted_disease}. Please provide prevention methods and possible medications."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input_message}]
    )
    response = completion.choices[0].message.content
    return response

# Function to generate a PDF report
def generate_pdf(disease, response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Disease Prevention and Medication Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Disease: {disease}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Prevention and Medications:\n{response}")
    return pdf.output(dest='S').encode('latin1')

# Function to create a download link for PDF
def create_download_link(pdf_data, filename):
    b64 = base64.b64encode(pdf_data).decode('latin1')
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF</a>'
    return href

# Streamlit UI setup
st.set_page_config(page_title="Disease Prediction and Assistance", page_icon="", layout="wide")

# Header
st.markdown("<h1 style='text-align: center;'>Disease Prediction and Assistance 🧑🏼‍⚕️</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <hr style="border: none; height: 1px; background: linear-gradient(90deg, #FF5733, #FFC300, #28B463);">
    """,
    unsafe_allow_html=True
)

# Input section for selecting symptoms
st.sidebar.header("Select Your Symptoms")
selected_symptoms = st.sidebar.multiselect("Choose symptoms from the list below:", symptom_names)

# Initialize session state variables
if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = None

# Main prediction and chatbot section
if st.sidebar.button("Predict Disease"):
    if selected_symptoms:
        # Create input vector
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_names]
        input_vector = np.array(input_vector).reshape(1, -1)
        input_vector_scaled = scaler.transform(input_vector)

        # Make prediction
        prediction = model.predict(input_vector_scaled)
        st.session_state.predicted_disease = label_classes[np.argmax(prediction)]
        st.session_state.chatbot_response = None  # Reset the chatbot response
    else:
        st.warning("Please select at least one symptom to proceed.")

# Display predicted disease
if st.session_state.predicted_disease:
    st.subheader(f"Predicted Disease: {st.session_state.predicted_disease}")

    # Assistance button
    if st.button("Get Assistance"):
        if st.session_state.chatbot_response is None:  # Generate response only once
            st.session_state.chatbot_response = generate_chatbot_response(st.session_state.predicted_disease)
        st.subheader("Prevention and Medications:")
        st.write(st.session_state.chatbot_response)

        # Generate PDF and create a download link
        pdf_data = generate_pdf(st.session_state.predicted_disease, st.session_state.chatbot_response)
        st.markdown(create_download_link(pdf_data, "disease_report.pdf"), unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <p style="text-align: center; font-size: 14px;">Powered by OpenAI API and Streamlit | Developed by Trio ❤️</p>
    """, unsafe_allow_html=True)
