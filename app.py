import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Mapowanie etykiet
label_mapping = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy', 
    4: 'Blueberry___healthy', 5: 'Cherry___healthy', 6: 'Cherry___Powdery_mildew', 
    7: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn___Common_rust', 9: 'Corn___healthy', 
    10: 'Corn___Northern_Leaf_Blight', 11: 'Grape___Black_rot', 12: 'Grape___Esca_(Black_Measles)', 
    13: 'Grape___healthy', 14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy', 
    18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight', 
    21: 'Potato___healthy', 22: 'Potato___Late_blight', 23: 'Raspberry___healthy', 
    24: 'Soybean___healthy', 25: 'Squash___Powdery_mildew', 26: 'Strawberry___healthy', 
    27: 'Strawberry___Leaf_scorch', 28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight', 
    30: 'Tomato___healthy', 31: 'Tomato___Late_blight', 32: 'Tomato___Leaf_Mold', 
    33: 'Tomato___Septoria_leaf_spot', 34: 'Tomato___Spider_mites Two-spotted_spider_mite', 
    35: 'Tomato___Target_Spot', 36: 'Tomato___Tomato_mosaic_virus', 
    37: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
}

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
    class_name = label_mapping.get(class_index, "Unknown Class")
    st.write(f"Predicted Class: {class_name}")
    st.write(f"Prediction Confidence: {probability:.2f}")
