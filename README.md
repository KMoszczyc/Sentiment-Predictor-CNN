# Sentiment Predictor
## About
Deep learning model trained on IMDB dataset to predict sentence sentiment, with around 89% accuracy. Ready for deployment on Heroku with Flask.

## Setup

#### Local
- run wsgi.py
- Open postman or any other program for testing API's.
- POST on endpoint:  http://127.0.0.1:5000/predict
`{
    "text": ["The acting is terrible, plot is boring and predictable. What a waste of time.."]
}`

## Model
![Screenshot](screenshots/model.png)
