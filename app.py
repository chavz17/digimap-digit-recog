import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
import gunicorn

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Load the model
model = tf.keras.models.load_model('digit_recognition_10.model')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def home():
    img = request.files['image']
    image_path = './static/files/' + img.filename
    img.save(image_path)

    # Preprocessing image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    _, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
    img = np.invert(np.array([img]))
    img = img.astype('float32') / 255.0

    # Get the prediction from the model
    prediction = model.predict(img)
    data = np.argmax(prediction)

    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=False, port=80)