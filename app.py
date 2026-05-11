import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from keras.src.ops.operation import Operation

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Fix HDF5 model deserialization when saved layer configs include
# unsupported `quantization_config` keys.
_orig_operation_from_config = Operation.from_config.__func__
@classmethod
def _patched_operation_from_config(cls, config):
    if isinstance(config, dict) and "quantization_config" in config:
        config = config.copy()
        config.pop("quantization_config")
    return _orig_operation_from_config(cls, config)
Operation.from_config = _patched_operation_from_config


app = Flask(__name__)
model = None
tokenizer = None

def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

@app.before_request
def before_first_request():
    load_keras_model()
    load_tokenizer()

def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]),2)

@app.route("/", methods=["GET", "POST"])
def index():
    text = None
    sentiment = None
    if request.method == "POST":
        text = request.form.get("user_text")
        if text:
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores(text)
            sentiment["custom model positive"] = sentiment_analysis(text)
    return render_template('form.html', sentiment=sentiment, user_text=text)
if __name__ == "__main__":
    app.run()
