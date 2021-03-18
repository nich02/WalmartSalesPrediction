import pandas as pd
from flask import Flask, jsonify, make_response, request, abort, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("savmodel.pkl", "rb"))

@app.route("/")
def hello():
  return render_template('index2.html')

@app.route("/get_prediction", methods=['POST','OPTIONS'])
#@cross_origin()
def get_prediction():
    if not request.json:
        abort(400)
    df = pd.DataFrame(request.json, index2=[0])
    cols=["Store","Dept","IsHoliday"]
    df = df[cols]
    return jsonify({'result': model.predict(df)[0]}), 201

if __name__ == "__main__":
  app.run()