import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = keras.models.load_model("modelpkl.h5")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

	feature1 = [float(x) for x in request.form.values('.closeprice')]
	feature2 = [float(x) for x in request.form.values('.compound')]
	feature1 = np.array(feature1)
	feature2 = np.array(feature2)

	final_features = pd.concat([feature1, feature2], axis=1)
	prediction = model.predict(final_features)


    # float_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(float_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Close Price for next day is ${}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)

