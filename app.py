from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('iris_decision_tree.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)
        prediction = model.predict(final_input)
        flower = ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]
        return render_template('index.html', prediction_text=f'Predicted Flower: {flower}')
    except:
        return render_template('index.html', prediction_text='Invalid input, please try again!')

if __name__ == "__main__":
    app.run(debug=True)
