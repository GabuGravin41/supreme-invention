import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = load_model('tea_disease_classifier.h5')

# Define class names (ensure these match the class indices used during training)
class_names = ['Algal leaf', 'Anthracnose', 'Bird eye spot', 'Brown blight', 'Gray light', 'Healthy', 'Red leaf spot', 'White spot']

# Streamlit interface
st.title("Tea Plant Disease Classifier")
st.write("Upload an image of a tea plant leaf to classify its disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    image = load_img(uploaded_file, target_size=(150, 150))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]

    # Display the image and prediction
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Class: {predicted_class_name}")
    st.write(f"Confidence: {np.max(predictions[0]) * 100:.2f}%")