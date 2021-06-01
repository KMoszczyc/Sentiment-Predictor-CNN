# Sentiment Predictor
## About
Deep learning model trained on IMDB dataset to predict sentence sentiment, with around 89% accuracy. Ready for deployment on Heroku with Flask.

## Usage

#### Local setup
- run wsgi.py
- Open postman or any other program for testing API's.
- POST on endpoint:  http://127.0.0.1:5000/predict

#### Examples
- body: `{
    "text": ["The acting is terrible, plot is boring and predictable. What a waste of time.."]
}`
- response: `[{"prediction":0.0,"probability":0.995847702}]`

- body: `{
    "text": ["Great acting, amazing cast. Movie is not trivial and not for an average viewer. Pushes to think and ponder on the meaning of life."]
}`
- response: `[{"prediction":1.0,"probability":0.9719628692}]`

## Model
![Screenshot](screenshots/model.png)
