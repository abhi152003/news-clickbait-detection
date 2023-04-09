from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import pickle

app = Flask(__name__)
CORS(app)

# Load the saved model
model = tf.keras.models.load_model('model8.h5',compile=False)

# Load the saved weights
model.load_weights('weights.h5')

# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def hello():
    return 'Hello, World!'

# Define the API route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the request body
    text = request.json['text']

    # Convert the text to a sequence using the loaded tokenizer
    sequence = tokenizer.texts_to_sequences([text])

    # Pad the sequence to a fixed length
    max_len = 500 # or whatever your sequence length is
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding='post')

    # Use the loaded model to make a prediction
    prediction = model.predict(padded)

    # Return the predicted label as a JSON response
    label = 'clickbait' if prediction[0][0] > 0.5 else 'not clickbait'
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run()
