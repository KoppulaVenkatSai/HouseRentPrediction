from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/main")
def page():
    return render_template("mainpage.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    forest_list = [int(x) for x in request.form.values()]
    linear_list = forest_list[0]

    my_dir = os.path.dirname(__file__)
    forest_model_path = os.path.join(my_dir, 'forestmodel.pkl')
    linear_model_path = os.path.join(my_dir, 'linearmodel.pkl')


    loaded_forest_model = pickle.load(open(forest_model_path, 'rb'))
    loaded_linear_model = pickle.load(open(linear_model_path, 'rb'))
    #[["BHK", "Size", "Area Type", "City", "Furnishing Status", "Tenant Preferred", "Bathroom"]]

    forest_final = np.array(forest_list)
    linear_final = np.array(linear_list)
    linear_final = linear_final.reshape(-1, 1)[0]

    forest_prediction = loaded_forest_model.predict([forest_final])
    linear_prediction = loaded_linear_model.predict([linear_final])

    output = (forest_prediction + linear_prediction)/2

    return render_template('mainpage.html',pred = "Predicted Rent :   Rs. " + str(int(output)))


if __name__ == "__main__":
    app.run(debug = True)
