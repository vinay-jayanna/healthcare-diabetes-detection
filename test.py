import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# Load trained model
model_path = "diabetic_retinopathy_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Define image path (Replace with the actual image path for testing)
test_image_path = "diabetic_retinopathy_data/test/0/21027_right.jpeg"  # Update with correct path

if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found: {test_image_path}")

# Load and preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input shape
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize using EfficientNet's preprocess
    return img_array

# Predict
img_array = preprocess_image(test_image_path)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]  # Get class with highest probability

# Display results
class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]  # Adjust labels if necessary
predicted_label = class_labels[predicted_class]

plt.imshow(image.load_img(test_image_path))
plt.axis("off")
plt.title(f"Predicted: {predicted_label}")
plt.show()

print(f"Prediction: {predicted_label} (Class {predicted_class})")
