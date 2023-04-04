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
    loc1 = request.form.get('loc1')
    loc2 = request.form.get('loc2')
    weight = request.form.get('weight')

    # initialize the geolocator
    geolocator = Nominatim(user_agent="my_app")

    # get the latitude and longitude of the first address
    location1 = geolocator.geocode(loc1)
    lat1, lon1 = location1.latitude, location1.longitude

    # get the latitude and longitude of the second address
    location2 = geolocator.geocode(loc2)
    lat2, lon2 = location2.latitude, location2.longitude

    print(loc1, loc2)
    print(location1)
    print(location2)

    # calculate the distance between the two locations using the Haversine formula
    dist = distance((lat1, lon1), (lat2, lon2)).km
    weight = request.form.get('weight')

    inputt_query = np.array([[dist, weight]])
    charges = charges_model.predict(inputt_query)[0]
    print(charges)

    return jsonify({'charges': str(round(charges))})

if __name__ == '__main__':
    #app.run(debug=True)
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
