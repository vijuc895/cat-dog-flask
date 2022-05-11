import os

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

# Load model
cnn_model = tf.keras.models.load_model("models/dog_cat_M.h5")

IMAGE_SIZE = 192


# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


# Predict & classify image
def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(
        preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model.predict(preprocessed_image)
    label = "Cat" if prob[0][0] >= 0.5 else "Dog"
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]

    return label, classified_prob


@app.route("/classify", methods=["POST"])
def upload_file():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)

    prob = round((prob * 100), 2)

    api_response = {
        "label": label,
        "confidence": prob
    }

    return api_response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)
