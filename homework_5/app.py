from flask import Flask, request, jsonify
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the model and the vectorizer
with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)

# Define a route to serve predictions
@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()

    # Transform input data using the dictionary vectorizer
    X = dv.transform([client_data])

    # Make a prediction
    prediction = model.predict_proba(X)[0, 1]  # Get probability for class 1 (subscription)

    # Return the prediction as JSON
    result = {"subscription_probability": prediction}
    return jsonify(result)

# Start the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)