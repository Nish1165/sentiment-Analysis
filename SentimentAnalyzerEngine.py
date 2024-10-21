import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Use raw string notation for the file path to avoid issues with backslashes
model_path = r'A:\B.Tech CSE\B.Tech sem - 5\Introduction to Big Data Tools and Techniques\Sentiment Analysis Project\Sentiment_Analysis_Case_Study\sentimentanalysis\models\model\classifier.pkl'

# Load the classifier model
try:
    classifier = joblib.load(model_path)
except FileNotFoundError:
    print(f"Model file not found at {model_path}")
    exit(1)

# Prediction function
def predictfunc(review):
    # Create a DataFrame from the input review
    df_review = pd.DataFrame({'Reviews': [review]})  # Use a DataFrame

    # Predict sentiment using the loaded classifier pipeline (which includes vectorizer and classifier)
    predicted_label = classifier.predict(df_review['Reviews'])  # Pass the DataFrame column to the pipeline

    # Extract the predicted label
    label = predicted_label[0]

    # Determine sentiment based on prediction
    if label == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    
    return label, sentiment

# Routes
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.form['review']  # Get the review from form input
        prediction, sentiment = predictfunc(content)  # Pass the review to the prediction function
        return render_template("predict.html", pred=prediction, sent=sentiment)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
