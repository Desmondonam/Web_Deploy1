from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get input data from POST request
    features = data['features']  # Adjust this based on your input data format

    # Make predictions using the loaded model
    prediction = model.predict([features])

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)