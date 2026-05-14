from flask import Flask, render_template, request
import os 
import sys
import numpy as np
import pandas as pd

# Force local src import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
	return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
	os.system("python main.py")
	return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
	if request.method == 'POST':
		try:
			#  reading the inputs given by the user
			try:
				fixed_acidity =float(request.form['fixed_acidity'])
				volatile_acidity =float(request.form['volatile_acidity'])
				citric_acid =float(request.form['citric_acid'])
				chlorides =float(request.form['chlorides'])
				total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
				density =float(request.form['density'])
				sulphates =float(request.form['sulphates'])
				alcohol =float(request.form['alcohol'])
			except ValueError:
				return render_template('index.html', error="Invalid input: Please enter numeric values only.")
	   
		 
			data = [fixed_acidity,volatile_acidity,citric_acid,chlorides,total_sulfur_dioxide,density,sulphates,alcohol]
			data = np.array(data).reshape(1, 8)
			
			obj = PredictionPipeline()
			predict = obj.predict(data)

			# Map numerical predictions to text labels
			result_map = {0: "Low Quality", 1: "Medium Quality", 2: "High Quality"}
			final_result = result_map.get(int(predict[0]), "Unknown")

			return render_template('results.html', prediction = final_result)

		except Exception as e:
			import traceback
			print('The Exception message is: ', e)
			print(traceback.format_exc())
			return 'something is wrong: ' + str(e)

	else:
		return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	
	# Debugging: Print file structure to see if artifacts are present
	print("--- Project Structure ---", flush=True)
	for root, dirs, files in os.walk("."):
		level = root.replace(".", "").count(os.sep)
		indent = " " * 4 * (level)
		print(f"{indent}{os.path.basename(root)}/", flush=True)
		subindent = " " * 4 * (level + 1)
		for f in files:
			print(f"{subindent}{f}", flush=True)
	print("-----------------------", flush=True)

	port = int(os.environ.get("PORT", 8080))
	app.run(host="0.0.0.0", port = port)
