import flask
from flask import request, jsonify
from content import load_tokenizer, load_model, pad_sequences
import pandas as pd
import numpy as np
import os

# disable it during training (it causes crashes when flask server runs)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.set_printoptions(precision=2, suppress=True)

app = flask.Flask(__name__)
app.config["DEBUG"] = True

tokenizer = load_tokenizer()
model = load_model('model_conv1d.h5')

@app.route('/predict', methods=['POST'])
def predictTextSentiment():
    content = request.json
    inputs = content['text'] if type(content['text']) == list else [content['text']]
    print(inputs)

    X_test = tokenizer.texts_to_sequences(inputs)
    X_test = pad_sequences(X_test, padding='post', maxlen=2000)
    probs = model.predict(x=X_test)
    predictions = np.round(probs, 0)
    probs = np.where(probs > 0.5, probs, 1-probs)

    results = pd.DataFrame({'prediction': predictions.flatten(), 'probability': probs.flatten()})
    print(results)
    return results.to_json(orient='records')
