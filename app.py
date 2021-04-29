from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import pandas as pd
from predict import predict

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


# @cross_origin()
'''class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = cotton(self.filename)'''


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/pred", methods=['POST'])
@cross_origin()
def predictRoute():
    if request.method == 'POST':
        airline = request.form['airline']
        source = request.form['source']
        destination = request.form['destination']
        duration = request.form['duration']
        totalstops = request.form['totalstops']
        addinfo = request.form['addinfo']
        date = request.form['date']
        month = request.form['month']
        arrivalh = request.form['ah']
        arrivalm = request.form['am']
        departureh = request.form['dh']
        departurem = request.form['dm']
        route1 = request.form['r1']
        route2 = request.form['r2']
        route3 = request.form['r3']
        route4 = request.form['r4']
        route5 = request.form['r5']

        predy = {'Airline': [airline], 'Source': [source], 'Destination': [destination], 'Duration': [duration],
                 'Total_Stops': [totalstops], 'Additional_Info': [addinfo], 'Date': [date], 'Month': [month],
                 'Arrival_Hour': [arrivalh], 'Arrival_Minute': [arrivalm],
                 'Departure_Hour': [departureh], 'Departure_Minute': [departurem], 'Route_1': [route1], 'Route_2': [route2],
                 'Route_3': [route3], 'Route_4': [route4],
                 'Route_5': [route5]}

        predf = pd.DataFrame(predy)
        result = predict(predf)

        return render_template('results.html',result=result)


#clApp = ClientApp()
# #port = int(os.getenv("PORT"))
if __name__ == "__main__":
    # clApp = ClientApp()
    # app.run(host='0.0.0.0', port=port)
    app.run(debug=True)
