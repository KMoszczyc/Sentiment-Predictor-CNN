from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Flatten, LSTM, GlobalMaxPooling1D, Embedding, Dense, Input, Dropout, Conv1D
from tensorflow.python.keras.models import Model
import numpy as np
from tensorflow.python.keras.models import load_model, save_model
import pandas as pd
import os
import re
import pickle

MAX_LEN = 2000


def load_dataset():
    """Load IMDB csv dataset"""

    imdb_dataset = pd.read_csv('datasets/IMDB Dataset.csv')
    imdb_dataset.replace({"positive": 1, "negative": 0}, inplace=True)
    imdb_dataset['review'] = imdb_dataset['review'].map(lambda sentence: preprocess_text(sentence))

    negative_words_dataset = load_words()
    dataset = pd.concat([imdb_dataset, negative_words_dataset], axis=0)

    dataset = dataset.sample(frac=1)
    print(dataset.head(10))
    return dataset


def load_words():
    """Load and preprocess positive and negative words"""

    X_neg = pd.read_csv('datasets/negative-words.txt', delimiter="\n")
    X_pos = pd.read_csv('datasets/positive-words.txt', delimiter="\n")
    X_neg.columns = ['review']
    X_pos.columns = ['review']
    temp_df = pd.DataFrame(X_neg['review'].map(lambda word: 'not ' + word), columns=['review'])
    X_pos = pd.concat([X_pos, temp_df], axis=0, ignore_index=True)

    X_neg['sentiment'] = [0] * len(X_neg)
    X_pos['sentiment'] = [1] * len(X_pos)

    dataset = pd.concat([X_pos, X_neg], axis=0)
    return dataset


def split_dataset(dataset, train_to_val_ratio=0.8):
    """Split the dataset to train and validation set"""

    dataset = dataset.sample(frac=1).reset_index(drop=True)
    val_set = dataset.iloc[int(len(dataset) * train_to_val_ratio):, :]
    train_set = dataset.iloc[:int(len(dataset) * train_to_val_ratio), :]

    return train_set, val_set


def preprocess_text(sen):
    """Remove numbers, enters, special signs from a given string"""

    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub("<br />", " ", sentence)
    sentence = sentence.lower()

    return sentence


def save_tokenizer(vectorizer, path):
    """Save the tokenizer"""

    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)


def load_tokenizer(path):
    """Load the tokenizer"""

    with open(path, 'rb') as f:
        vectiorizer = pickle.load(f)
        return vectiorizer


def create_model(tokenizer):
    """Create a Keras model with embedding, 3 1D conv layers and 2 dense layers"""

    max_features = len(tokenizer.word_index) + 1
    embedding_dim = 64

    deep_inputs = Input(shape=(MAX_LEN,))
    x = Embedding(max_features, embedding_dim)(deep_inputs)
    x = Dropout(0.5)(x)
    x = Conv1D(64, 8, padding="valid", activation="relu", strides=3)(x)
    x = Conv1D(128, 8, padding="valid", activation="relu", strides=3)(x)
    x = Conv1D(256, 8, padding="valid", activation="relu", strides=3)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(deep_inputs, predictions)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train():
    """Train the neural network and save the model"""

    dataset = load_dataset()
    print('max: ', len(max(dataset['review'])))
    print('mean :', dataset.review.str.len().mean())

    train_set, val_set = split_dataset(dataset, 0.9)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_set['review'])

    X_train = tokenizer.texts_to_sequences(train_set['review'])
    X_val = tokenizer.texts_to_sequences(val_set['review'])

    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    X_train = pad_sequences(X_train, padding='post', maxlen=MAX_LEN)
    X_val = pad_sequences(X_val, padding='post', maxlen=MAX_LEN)

    model = create_model(tokenizer)

    history = model.fit(x=X_train, y=np.array(train_set['sentiment']),
                        validation_data=(X_val, np.array(val_set['sentiment'])),
                        batch_size=128,
                        epochs=3,
                        verbose=1)

    print(history.history['val_accuracy'])

    save_model(model, 'models/model_conv1d.h5')
    save_tokenizer(tokenizer, 'models/tokenizer.pickle')


def test():
    """Test the neural network accuracy"""

    np.set_printoptions(suppress=True)
    test_reviews = [
        # Positive Reviews
        'This is an excellent movie',
        'The movie was fantastic I like it',
        'You should watch it is brilliant',
        'Exceptionally good',
        'Wonderfully directed and executed I like it',
        'Its a fantastic series',
        'Never watched such a brilliant movie',
        'It is a Wonderful movie',

        # Negtive Reviews
        "horrible acting",
        'waste of money',
        'pathetic picture',
        'It was very boring',
        'I did not like the movie',
        'The movie was horrible',
        'I will not recommend',
        'The acting is pathetic',
        'This movie is for morons'
    ]

    test_reviews = map(lambda sentence: preprocess_text(sentence), test_reviews)
    y_test = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    tokenizer = load_tokenizer('models/tokenizer.pickle')
    model = load_model('models/model_conv1d.h5')
    X_test = tokenizer.texts_to_sequences(test_reviews)
    X_test = pad_sequences(X_test, padding='post', maxlen=2000)
    predictions = model.predict(x=X_test)
    print(predictions)

    model.evaluate(x=X_test, y=y_test)
