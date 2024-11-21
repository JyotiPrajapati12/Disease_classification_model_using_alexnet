import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras
from keras import preprocessing
import os
import json
from zipfile import ZipFile
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

#kaggle_credentails = json.load(open("C:\Users\93in\plant-disease\kaggle.json"))
# Dataset Path
base_dir = '/plant-disease/plantvillage dataset/plantvillage dataset/color'
# Image Parameters
img_size = 224
batch_size = 32
# Image Data Generators
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% of data for validation
)
# Train Generator
# Train Generator
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(227, 227),  # Changed from 224x224 to 227x227
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)
# Validation Generator
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(227, 227),  # Changed from 224x224 to 227x227
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)


# Define the AlexNet architecture
model = Sequential([
    # First convolutional layer
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    BatchNormalization(),

    # Second convolutional layer
    Conv2D(256, (5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    BatchNormalization(),

    # Third, Fourth, and Fifth convolutional layers
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    BatchNormalization(),

    # Flatten and Fully Connected layers
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),

    # Output layer (adjust the number of units to match your number of classes)
    Dense(len(train_generator.class_indices), activation='softmax')  # Use the number of classes in your dataset
])

# Display the model summary
model.summary()

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,       # Number of steps per epoch
    epochs=5,                                                    # Number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # Validation steps
)

# Accessing the training and validation accuracy from the history
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Display the accuracy for each epoch
for epoch in range(len(training_accuracy)):
    print(f"Epoch {epoch + 1}:")
    print(f"Training Accuracy: {training_accuracy[epoch]:.4f}")
    print(f"Validation Accuracy: {validation_accuracy[epoch]:.4f}\n")

# Final accuracy
print(f"Final Training Accuracy: {training_accuracy[-1]:.4f}")
print(f"Final Validation Accuracy: {validation_accuracy[-1]:.4f}")
# Plotting the accuracy graph
plt.figure(figsize=(10, 6))

# Training accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')

# Validation accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

# Graph labels and legend
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


#model.save('plant_disease_classification_model12.h5')
