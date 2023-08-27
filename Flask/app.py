from flask import Flask, request, jsonify, render_template, request
import joblib
import numpy as np


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set (optional)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(model, 'iris_model.pkl')

# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

# Load the trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    features = [float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])]

    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    predicted_class = iris.target_names[prediction][0]

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)