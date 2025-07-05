from flask import Flask, render_template, request, redirect, url_for
import joblib, os
from preprocessing import preprocess

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))


# 👉 Halaman prediksi
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    pred = None
    if request.method == 'POST':
        review = request.form['review']
        vec = vectorizer.transform([preprocess(review)])
        pred = model.predict(vec)[0]
    return render_template('index.html', prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)
