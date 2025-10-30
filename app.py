from flask import Flask, render_template, jsonify
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import random

app = Flask(__name__)

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Labels
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=1, verbose=1)  # train quickly

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    i = random.randint(0, len(x_test)-1)
    img = x_test[i].reshape(1, 28, 28, 1)
    pred = np.argmax(model.predict(img))
    return jsonify({'prediction': labels[pred]})

if __name__ == '__main__':
    app.run(debug=True)
