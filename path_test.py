import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess test image
def preprocess_test_image(image_path, image_size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    # Load and resize the image
    img = load_img(image_path, target_size=image_size)
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Predict the cyclone path
def predict_cyclone_path(model_path, test_image_path, image_size=(128, 128)):
    # Load the trained model
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Preprocess the test image
    print(f"Preprocessing test image: {test_image_path}...")
    test_image = preprocess_test_image(test_image_path, image_size)

    # Predict the future path
    print("Predicting future path...")
    predicted_path = model.predict(test_image)

    print("Prediction completed.")
    return predicted_path

# Plot predicted cyclone path
def plot_cyclone_path(predicted_path):
    print("Plotting cyclone path...")
    # Extract latitude and longitude from predicted path
    latitudes = predicted_path[0][:, 1]
    longitudes = predicted_path[0][:, 2]

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(longitudes, latitudes, marker='o', color='b', label="Predicted Path")
    plt.title('Predicted Cyclone Path (Latitude vs Longitude)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    model_path = "cyclone_path_model_update.h5"  # Path to the saved model
    test_image_path = "./images/s2.jpg"  # Path to the test image (replace with your image path)

    # Predict the cyclone path and plot the result
    try:
        predicted_path = predict_cyclone_path(model_path, test_image_path)
        plot_cyclone_path(predicted_path)
    except Exception as e:
        print(f"Error: {e}")
