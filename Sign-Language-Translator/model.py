# model.py

import tensorflow as tf
import numpy as np

class KeyPointClassifier:
    def __init__(self, model_path='D:\\new sign\\Sign-Language-Translator\\model\\keypoint_classifier\\model.hdf5'):
        # Load the trained model (adjust the path to your model)
        self.model = tf.keras.models.load_model(model_path)

    def __call__(self, landmarks):
        """
        This method should preprocess the input landmarks and return the model's prediction
        """
        input_data = np.array(landmarks).reshape(1, -1)  # Example: reshape for Keras model
        prediction = self.model.predict(input_data)
        return np.argmax(prediction)  # Assuming classification with a single output class
