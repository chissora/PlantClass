import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model_vgg16_v2.keras")

# Load the model
model = load_model()

# Function to preprocess the image and predict
def preprocess_and_predict(image, model):
    # Process the image as required by the model
    image = image.resize((128, 128))  # Resize to 128x128
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    predictions = model.predict(image_array)  # Make predictions
    class_index = np.argmax(predictions)  # Get the predicted class index
    probability = predictions[0][class_index]  # Get the prediction confidence
    return class_index, probability

# Streamlit interface
st.title("Plant Disease Recognizer")
st.subheader("Identify plant diseases using a VGG16-based model")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing the image and making predictions...")

    # Make predictions
    class_index, probability = preprocess_and_predict(image, model)

    # Display the results
    st.write(f"Predicted Class Index: {class_index}")
    st.write(f"Prediction Confidence: {probability:.2f}")
