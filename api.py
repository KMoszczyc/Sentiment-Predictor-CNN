import flask
from flask import request, jsonify
from content import load_tokenizer, load_model, pad_sequences
import pandas as pd
import numpy as np
import os

# set to -1 for deployment (it causes crashes when flask server runs)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.set_printoptions(precision=2, suppress=True)

app = flask.Flask(__name__)
app.config["DEBUG"] = True

tokenizer = load_tokenizer('models/tokenizer.pickle')
model = load_model('models/model_conv1d.h5')


@app.route('/predict', methods=['POST'])
def predict_text_sentiment():
    """POST endpoint for text sentiment prediction. If you run this locally POST on http://127.0.0.1:5000/predict.

    Example json input (1 sentence):
    { "text": ["The acting is terrible, plot is boring and predictable. What a waste of time.."] }

    or multiple sentences:
    { "text": ["The acting is terrible, plot is boring and predictable. What a waste of time..", "A very nice movie",
        "I like drinking beer at the sunset."] }

    Returns:
        A JSON with prediction value, 0 - negivative sentiment, 1 - positive and a probability [0; 1]
        example:
            [{"prediction":0.0,"probability":0.995847702}]
    """

    content = request.json
    inputs = content['text'] if type(content['text']) == list else [content['text']]
    print(inputs)

    X_test = tokenizer.texts_to_sequences(inputs)
    X_test = pad_sequences(X_test, padding='post', maxlen=2000)
    probs = model.predict(x=X_test)
    predictions = np.round(probs, 0)
    probs = np.where(probs > 0.5, probs, 1 - probs)

    results = pd.DataFrame({'prediction': predictions.flatten(), 'probability': probs.flatten()})
    print(results)
    return results.to_json(orient='records')
