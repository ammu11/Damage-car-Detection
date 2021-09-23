import os
import sys

import uvicorn as uvicorn
from flask import Flask, render_template, request
from object_detector_detection_api_lite import *

import flask
from werkzeug.utils import secure_filename

import tensorflow as tf
from PIL import Image
import PIL
import numpy as np
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp'
host = "0.0.0.0"

@app.route('/')
def welcome():
    return render_template("login.html")
	
@app.route('/index')
def welcome1():
    return render_template("index.html")	

@app.route('/uploads')
def uploads():
    page = request.args
    bot = dict(page)["variable"]
    print(bot)
    return render_template("upload.html",bot="/"+bot)
	

@app.route('/CarDamage', methods=['GET', 'POST'])
def upload_file2():
  
	file = []
	file = flask.request.files.getlist("file[]")
	print("****:",file)
	f = len(file)
	output = {}
	if f > 1:

		for i in file:  #i is file storage
			filename = secure_filename(i.filename) # save file 
			filepath = "out"+"/"+filename
			i.save(filepath)
			me=main(filename)
			filename = secure_filename(i.filename)
			me1=completenpartial(i)
			output[filename] = "<b>"+"The car is "+" "+"<font color=Crimson>"+me1+"</font>"+" "+" and the damages of the car are"+" "+"<font color=Crimson>"+str(me).replace("[","").replace("]","").replace("'","")+"</font>"+"</b>"
		columns = [{"field": "Image FileName", "title": "Image FileName","sortable": True},{"field": "Damage Recognition","title": "Car Damage Parts","sortable": True}]
		res = []
		for key,val in output.items():
			b = {columns[0]["field"]:key,columns[1]["field"]:val}
			res.append(b)

		return render_template("table1.html",
		data=res,
		columns=columns,
		title='Damage Prediction Results')
	else:
		file = flask.request.files["file"]
		# model = load_model(os.getcwd()+'/model/my_model2.h5')
		filename = secure_filename(file.filename) # save file 
		filepath = "out"+"/"+filename
		file.save(filepath)
		me=main(filename)
		me1=completenpartial(file)
		filename = secure_filename(file.filename)
		output[filename] = "<b>"+"The car is "+" "+"<font color=Crimson>"+me1+"</font>"+" "+" and the damages of the car are"+" "+"<font color=Crimson>"+str(me).replace("[","").replace("]","").replace("'","")+"</font>"+"</b>"
		return "The car is"+" "+me1+" "+" and the damages of the car are"+" "+str(me).replace("[","").replace("]","").replace("'","")
def completenpartial(file):
	interpreter = tf.lite.Interpreter(model_path="models/fullnpartial_size_224.tflite")
	interpreter.allocate_tensors()
	x,y=0,0
	img = Image.open(file)
	img = img.resize((224,224), PIL.Image.ANTIALIAS)

	# Normalize to [0, 1]
	data = np.asarray( img, dtype="int32" ) / 255.0

	# Inference on input data normalized to [0, 1]
	inputImg = np.expand_dims(data,0).astype(np.float32)
	input_details = interpreter.get_input_details()
	interpreter.set_tensor(input_details[0]['index'], inputImg)
	interpreter.invoke()

	output_details = interpreter.get_output_details()
	output_data = interpreter.get_tensor(output_details[0]['index'])


	if(format(np.argmax(output_data))=="1"):
		me1="Partially Captured"
	else:
		me1="Completely Captured"

	return me1
@app.route('/frontnback', methods=['GET', 'POST'])
def upload_file3():

	interpreter = tf.lite.Interpreter(model_path="models/frontnback_size_299.tflite")
	interpreter.allocate_tensors()
	x,y=0,0
	  
	file = []
	file = flask.request.files.getlist("file[]")   
	f = len(file)
	# print("/////////////////",f)
	output = {}
	if f > 1:
		for i in file:
			img = Image.open(i)
			img = img.resize((299,299), PIL.Image.ANTIALIAS)

			# Normalize to [0, 1]
			data = np.asarray( img, dtype="int32" ) / 255.0

			# Inference on input data normalized to [0, 1]
			inputImg = np.expand_dims(data,0).astype(np.float32)
			input_details = interpreter.get_input_details()
			interpreter.set_tensor(input_details[0]['index'], inputImg)
			interpreter.invoke()

			output_details = interpreter.get_output_details()
			output_data = interpreter.get_tensor(output_details[0]['index'])


			if(format(np.argmax(output_data))=="1"):
				me="Back Side"
			else:
				me="Front Side"
			
			filename = secure_filename(i.filename)
			s= filename.rsplit("_", 1)[1]
			#filename = os.path.join(app.config['UPLOAD_FOLDER'],filename)
			output[s] = "<b>"+"The car location is facing"+" "+"<font color=Crimson>"+me+"</font>"+"</b>"
			columns = [{"field": "Image FileName", "title": "Image FileName","sortable": True},{"field": "Front or Back Recognition","title": "Front or Back Recognition","sortable": True}]
			res = []
			for key,val in output.items():
				b = {columns[0]["field"]:key,columns[1]["field"]:val}
				print(b)
				res.append(b)
		print(res)
		return render_template("table1.html",
		data=res,
		columns=columns,
		title='Front and Back Detection')
	else:
		file = request.files['file']
		print("//////////////////////",file)
		# model = load_model('E:/BMW_IMAGE_ANALYSIS/model/newmodelcarsvsbmw.h5')
		# save the model to disk
		output = {}
		img = Image.open(file)
		img = img.resize((299,299), PIL.Image.ANTIALIAS)

		# Normalize to [0, 1]
		data = np.asarray( img, dtype="int32" ) / 255.0

		# Inference on input data normalized to [0, 1]
		inputImg = np.expand_dims(data,0).astype(np.float32)
		input_details = interpreter.get_input_details()
		interpreter.set_tensor(input_details[0]['index'], inputImg)
		interpreter.invoke()

		output_details = interpreter.get_output_details()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		if(format(np.argmax(output_data))=="1"):
			me="Back Side"
		else:
			me="Front Side"
	  
		filename = secure_filename(file.filename)
		output[filename] = "The car location is facing"+" "+me
		return("The car location is facing"+" "+me)	

	

@app.route('/leftnright', methods=['GET', 'POST'])
def upload_file():
	interpreter = tf.lite.Interpreter(model_path="models/leftnright_size_224.tflite")
	interpreter.allocate_tensors()
	x,y=0,0
	  
	file = []
	file = flask.request.files.getlist("file[]")   
	f = len(file)
	output = {}
	if f > 1:
		for i in file:
			img = Image.open(i)
			img = img.resize((224,224), PIL.Image.ANTIALIAS)

			# Normalize to [0, 1]
			data = np.asarray( img, dtype="int32" ) / 255.0

			# Inference on input data normalized to [0, 1]
			inputImg = np.expand_dims(data,0).astype(np.float32)
			input_details = interpreter.get_input_details()
			interpreter.set_tensor(input_details[0]['index'], inputImg)
			interpreter.invoke()

			output_details = interpreter.get_output_details()
			output_data = interpreter.get_tensor(output_details[0]['index'])


			if(format(np.argmax(output_data))=="1"):
				me="Right Side"
			else:
				me="Left Side"
			
			filename = secure_filename(i.filename)
			s= filename.rsplit("_", 1)[1]
			#filename = os.path.join(app.config['UPLOAD_FOLDER'],filename)
			output[s] = "<b>"+"The car location is facing"+" "+"<font color=Crimson>"+me+"</font>"+"</b>"

			columns = [{"field": "Image FileName", "title": "Image FileName","sortable": True},{"field": "Left or Right Recognition","title": "Left or Right Recognition","sortable": True}]
			res = []
			for key,val in output.items():
				b = {columns[0]["field"]:key,columns[1]["field"]:val}
				res.append(b)

		return render_template("table1.html",
		data=res,
		columns=columns,
		title='Left and Right Detection')
	else:
		file = request.files['file']
		# model = load_model('E:/BMW_IMAGE_ANALYSIS/model/newmodelcarsvsbmw.h5')
		# save the model to disk
		output = {}
		img = Image.open(file)
		img = img.resize((224,224), PIL.Image.ANTIALIAS)

		# Normalize to [0, 1]
		data = np.asarray( img, dtype="int32" ) / 255.0

		# Inference on input data normalized to [0, 1]
		inputImg = np.expand_dims(data,0).astype(np.float32)
		input_details = interpreter.get_input_details()
		interpreter.set_tensor(input_details[0]['index'], inputImg)
		interpreter.invoke()

		output_details = interpreter.get_output_details()
		output_data = interpreter.get_tensor(output_details[0]['index'])
		
		if(format(np.argmax(output_data))=="1"):
			me="Right Side"
		else:
			me="Left Side"
			
		filename = secure_filename(file.filename)
		output[filename] = "The car location is facing"+" "+me
		return(output[filename])	

#
# if __name__ == '__main__':
#     app.run(host=host, port=port)
if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008))
        uvicorn.run(app, host = "0.0.0.0", port = port)