import tensorflow as tf
from utils.preprocessing import load_data

def evaluate_model():
    """Loads the trained model and evaluates it on test data."""
    X_train, X_test, y_train, y_test, _, _ = load_data()
    model = tf.keras.models.load_model("models/sign_model.h5")

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"📊 Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
