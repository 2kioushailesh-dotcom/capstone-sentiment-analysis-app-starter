from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    text = None
    sentiment = None
    if request.method == "POST":
        text = request.form.get("user_text")
        if text:
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores(text)
    return render_template('form.html', sentiment=sentiment, user_text=text)
if __name__ == "__main__":
    app.run()
