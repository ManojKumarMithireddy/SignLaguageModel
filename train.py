import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from utils.preprocessing import load_data

def create_model(input_shape, num_classes):
    """Defines the CNN model architecture."""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Loads data, trains the model, and saves it."""
    X_train, X_test, y_train, y_test, input_shape, num_classes = load_data()
    model = create_model(input_shape, num_classes)
    
    print("📌 Training started...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    model.save("models/sign_model.h5")
    print("✅ Model saved as sign_model.h5")

if __name__ == "__main__":
    train_model()
