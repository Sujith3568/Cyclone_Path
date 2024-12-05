import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RepeatVector, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# Load data
def load_data(image_dir, label_csv_path, image_size=(128, 128)):
    print("Loading data...")
    
    # Load labels
    label_data = pd.read_csv(label_csv_path)
    images = []
    labels = []

    for _, row in label_data.iterrows():
        image_path = os.path.join(image_dir, f"{row['id']}.jpg")
        if os.path.exists(image_path):
            # Load and preprocess image
            img = load_img(image_path, target_size=image_size)  # Resize to target size
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
            
            # Extract label (path as 2D array)
            path = eval(row['additional_points'])  # Convert string representation to list
            labels.append(path)

    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return images, labels

# Preprocess data
def preprocess_data(images, labels, max_future_steps=10):
    print("Preprocessing data...")
    X = images
    y = []

    for label in labels:
        if len(label) > max_future_steps:
            y.append(label[:max_future_steps])  # Truncate to max_future_steps
        else:
            # Pad with the last point if path is shorter
            pad = np.tile(label[-1], (max_future_steps - len(label), 1))
            y.append(np.vstack((label, pad)))

    y = np.array(y)
    print(f"Processed data shapes - X: {X.shape}, y: {y.shape}")
    return X, y

# Build model
def build_model(input_shape, output_sequence_length):
    print("Building model...")
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)  # Regularization
    x = RepeatVector(output_sequence_length)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)  # Regularization
    outputs = Dense(2, activation='linear')(x)  # Predict (latitude, longitude)
    model = Model(inputs, outputs)
    
    # Explicitly define loss and metrics
    model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    
    print("Model built successfully.")
    return model

# Train and save model
def train_and_save_model(image_dir, label_csv_path, model_save_path, image_size=(128, 128), epochs=20, batch_size=32):
    images, labels = load_data(image_dir, label_csv_path, image_size)
    max_future_steps = 10
    X, y = preprocess_data(images, labels, max_future_steps)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:]  # Shape of each image
    model = build_model(input_shape, max_future_steps)

    print("Starting training...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    print("Training complete.")

    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.")

# Main function
if __name__ == "__main__":
    image_dir = "insat3d_ir_cyclone_ds\\CYCLONE_DATASET_INFRARED"
    label_csv_path = "labeled_cyclone_data.csv"
    model_save_path = "cyclone_path_model_update.h5"

    train_and_save_model(image_dir, label_csv_path, model_save_path)
