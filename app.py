import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(227, 227)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # Convert the index to a string to match the dictionary keys
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Load the trained model
model = load_model(r'C:\Users\93in\OneDrive\Desktop\plant_disease_classification_CS_project\plant_disease_classification_model.h5')

# Load class indices
class_indices = {
    "0": "Apple___Apple_scab", 
    "1": "Apple___Black_rot", 
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy", 
    "4": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "5": "Corn_(maize)___Common_rust_",
    "6": "Corn_(maize)___healthy"
}

# Load the disease information JSON
with open('C:/Users/93in/plant-disease/disease.json') as f:
    disease_data = json.load(f)

# Streamlit Interface
st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to predict its disease and get detailed information.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Save the image to a temporary path
    temp_path = "temp_image.jpg"
    img.save(temp_path)

    # Predict the disease
    predicted_class_name = predict_image_class(model, temp_path, class_indices)

    # Process the predicted class name to extract disease info
    parts = predicted_class_name.split("___")
    after_delimiter = parts[-1]
    disease = after_delimiter.rstrip('_').replace("_", " ")

    # Display the prediction
    if disease == "healthy":
        st.success("The leaf is healthy!")
    else:
        st.error(f"The predicted disease is: {disease}")

    # Display the image again for context
    #img = mpimg.imread(temp_path)
    #fig, ax = plt.subplots()
    #ax.imshow(img)
    #ax.axis('off')  # Hide axes
    #st.pyplot(fig)
