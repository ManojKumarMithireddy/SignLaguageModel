import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "data/"
IMG_SIZE = 128  # Resize images to 128x128

def load_data():
    """Loads and preprocesses images from the dataset directory."""
    classes = os.listdir(DATASET_PATH)
    X, y = [], []

    for label in classes:
        class_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(class_path):
            continue  # Skip non-folder files
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X) / 255.0  # Normalize pixel values
    y = pd.factorize(y)[0]  # Convert labels to numerical values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = len(classes)

    return X_train, X_test, y_train, y_test, input_shape, num_classes
