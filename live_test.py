import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/sign_model.h5")

# Define image size (should match training size)
IMG_SIZE = 128  

# Define labels (Make sure this matches your dataset)
LABELS = ["Sign 1", "Sign 2", "Sign 3", "Sign 4"]  # Update with actual class names

# Open webcam
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict using the model
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    predicted_label = LABELS[class_index]  

    # Display prediction
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Sign Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
