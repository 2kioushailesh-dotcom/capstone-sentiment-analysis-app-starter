import pickle
from tensorflow.keras.preprocessing import sequence
import pytest
import app

@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    return app.app.test_client()

def load_tokenizer():
    with open('models/tokenizer.pickle', 'rb') as handle:
        app.tokenizer = pickle.load(handle)

def test_load_keras_model():
    app.load_keras_model()
    assert app.model is not None

def test_load_tokenizer():
    load_tokenizer()
    assert app.tokenizer is not None

def test_sentiment_analysis():
    input_text = "I love python"
    expected_output = 0.9
    app.load_keras_model()
    load_tokenizer()
    user_sequences = app.tokenizer.texts_to_sequences([input_text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = app.model.predict(user_sequences_matrix)
    assert prediction[0][0] >= expected_output

def test_index_get(client):
    response = client.get('/')
    assert response.status_code == 200

def test_index_post_positive_sentiment(client):
    response = client.post('/', data={'user_text': 'I love python'})
    assert response.status_code == 200
    assert b"Positive" in response.data

def test_index_post_negative_sentiment(client):
    response = client.post('/', data={'user_text': 'I hate python'})
    assert response.status_code == 200
    assert b"Negative" in response.data