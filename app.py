with open ('app.py','w') as f:
  f.write("""
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('/content/resnet50_grayscale.h5')

# Define your custom class labels
class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Function to preprocess uploaded image
def preprocess_image(image_file):
    # Load the grayscale image
    img = image.load_img(image_file, target_size=(224, 224), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # normalize pixel values to the range [0, 1]
    return img

# Custom decode_predictions function for your custom classes
def custom_decode_predictions(preds, top=1):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_labels[i], pred[i]) for i in top_indices]
        results.append(result)
    return results

# Main function to run the Streamlit web app
def main():
    st.title("Disease Classification")
    st.write("Upload a grayscale image for classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = preprocess_image(uploaded_file)
        st.image(img[0], caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            prediction = model.predict(img)
            decoded_preds = custom_decode_predictions(prediction, top=1)[0]
            predicted_class_label = decoded_preds[0][0]
            confidence = decoded_preds[0][1]
            st.write(f"Prediction: {predicted_class_label}, Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()


    """)
