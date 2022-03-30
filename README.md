# Sentiment Predictor
## About
Deep learning model trained on IMDB dataset to predict sentence sentiment with around 89% accuracy. The model was also trained on a list of popular positive and negative words becouse many of those are missing from IMDB dataset (duh).
Ready for deployment on Heroku with Flask.

## Usage

#### Local setup
- run wsgi.py
- Open postman or any other program for testing API's.
- POST on endpoint:  http://127.0.0.1:5000/predict

#### or just use my deployed model on Heroku
- make POST request on endpoint: https://sentiment-prediction-deepl.herokuapp.com/predict with json that looks like some of those below

#### Examples
(prediction 0 - negative sentence, 1 - positive sentence)

`{
    "text": ["The acting is terrible, plot is boring and predictable. What a waste of time.."]
}`

response: `[{"prediction":0.0,"probability":0.995847702}]`  

 `{
    "text": ["Great acting, amazing cast. Movie is not trivial and not for an average viewer. Pushes to think and ponder on the meaning of life."]
}`

response: `[{"prediction":1.0,"probability":0.9719628692}]`  


`{
    "text": ["The pancakes were out of this world, I've never eaten something so tasty in my life"]
}`

response: `[{"prediction":1.0,"probability":0.9649505019}]`  

`{
    "text": ["Staff doesn't care about the customer, had to wait for 30 minutes till somebody showed up. Huge disappointment."]
}`

response: `[{"prediction":0.0,"probability":0.8519150615}]`  

#### Or you can send multiple text samples
`{ "text": ["The acting is terrible, plot is boring and predictable. What a waste of time..", "A very nice movie", "I like drinking beer at the sunset."] }`  
response: `[{"prediction":0.0,"probability":0.995847702},{"prediction":1.0,"probability":0.8738321066},{"prediction":1.0,"probability":0.9644991755}]`
## Model
![Screenshot](screenshots/model.png)
