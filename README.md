# Plant Disease Detection and Classification using Deep Learning

This project implements a web-based application for plant disease detection and classification. Using a convolutional neural network (AlexNet) architecture, it identifies various plant diseases from uploaded leaf images. The web interface, built using Streamlit, provides a user-friendly platform for uploading images and viewing results.

---

## Features

- **Image Classification**: Detects and classifies diseases in plant leaf images.
- **Pretrained Model**: Leverages a trained AlexNet-inspired model for robust classification.
- **Web Interface**: User-friendly interface to upload images and receive predictions.
- **Disease Information**: Provides detailed information about the detected disease, helping users understand treatment options.

---

## Tools and Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow and Keras
- **Web Framework**: Streamlit
- **Data Preprocessing**: NumPy, PIL, and Matplotlib
- **Dataset**: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

pip install -r requirements.txt

## Project Structure
```bash

plant-disease-detection/
├── app.py                   # Streamlit application script
├── model.py                 # Script to train the AlexNet model
├── plant_disease_classification_model.h5  # Trained model
├── plantvillage dataset/    # Dataset directory
├── disease.json             # JSON file containing disease details
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
