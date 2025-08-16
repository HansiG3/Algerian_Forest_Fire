import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
## request:An object that contains data sent from the client (browser, API call, etc.) to your server.
##jsonify:A helper function that converts Python data (dicts, lists, etc.) into JSON format and sends it as a response.
##render_template:A function that loads an HTML file from your templates folder and sends it to the browser.
##Why you need it: Lets you make dynamic web pages by combining HTML with Python variables.

application=Flask(__name__)
##__name__ is a special Python variable.
##If you are running a file directly (e.g., python app.py), __name__ will equal "__main__".
##If the file is being imported from somewhere else, __name__ will be the name of the module.
 
app=application

##import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        month = float(request.form.get('month'))
        Temperature=float(request.form.get('Temperature')) ##after.get name should be same as mentioned in form name attribute
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[month,Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',result=result[0]) ##result is a list

    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
