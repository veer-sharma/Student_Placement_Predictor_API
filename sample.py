from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
charges_model = pickle.load(open('charges.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello Veer!'

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    input_query = np.array([[cgpa, iq, profile_score]])
    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})

@app.route('/calculate', methods=['POST'])
def calculate():
    dist = request.form.get('dist')
    weight = request.form.get('weight')

    inputt_query = np.array([[dist, weight]])
    charges = charges_model.predict(inputt_query)[0]
    print(charges)

    return jsonify({'charges': str(round(charges))})

if __name__ == '__main__':
    #app.run(debug=True)
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
