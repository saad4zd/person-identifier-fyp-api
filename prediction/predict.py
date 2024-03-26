import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import top_k_categorical_accuracy


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(
        128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict():
    loaded_model = load_model(os.path.join(os.getcwd(
    ), 'model/efficientnetb0.h5'), custom_objects={'top_5_accuracy': top_5_accuracy})

    image_path = os.path.join(os.getcwd(), 'gei', 'gei.png')
    preprocessed_image = preprocess_image(image_path)
    predictions = loaded_model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    return predicted_class_index+1, confidence
