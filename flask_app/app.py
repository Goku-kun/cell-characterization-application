from flask.helpers import url_for
from inference import infer
from functions import is_authenticated, validate_file
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import os
import urllib.request
import hashlib as hash
from datetime import datetime

upload_dir = './uploads/'
allowed_file_types = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = '-r-.gone.on-my-way-home.*X5'
app.config['UPLOAD_FOLDER'] = upload_dir
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

storage = dict()

@app.route('/upload', methods=['GET', 'POST'])
def upload():
   # authenticate
   if not is_authenticated(request.headers.get('x-api-key')):
      response = jsonify({'message' : 'You are not authorized to access this resource'})
      response.status_code = 400
      return response
   # check if the post request has the file part
   if 'file' not in request.files:
      response = jsonify({'message' : 'No file part in the request'})
      response.status_code = 400
      return response
   
   file = request.files['file']

   # check if the file is empty
   if file.filename == '':
      response = jsonify({'message' : 'No file selected for uploading'})
      response.status_code = 400
      return response

   # validate file type
   if file and validate_file(file.filename, allowed_file_types):
      filename = secure_filename(file.filename)
      to_hash = (filename + str(datetime.now())).encode('utf-8')
      image_id = str(hash.sha1(to_hash).hexdigest().encode('utf-8'))[2:-1]
      save_as = image_id + f'.{filename.split(".")[1]}'
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], save_as))
      storage[image_id] = save_as
      print(f'Stored {save_as} in Storage')
      response = jsonify({'message' : 'File successfully uploaded', 'image_id': image_id})
      response.status_code = 201
      return response
   else:
      response = jsonify({'message' : 'Invalid file type. Allowed file types are png, jpg and jpeg'})
      response.status_code = 400
      return response

@app.route('/process', methods=['GET', 'POST'])
def process():
   # authenticate
   if not is_authenticated(request.headers.get('x-api-key')):
      response = jsonify({'message' : 'You are not authorized to access this resource'})
      response.status_code = 400
      return response
   # check for id
   if 'image_id' not in request.headers:
      print(request.headers.keys())
      print(request.headers.values())
      response = jsonify({'message' : 'Required param image_id not found'})
      response.status_code = 400
      return response

   # check if image is in storage
   image_id = request.headers.get('image_id')
   if not image_id in storage:
      response = jsonify({'message' : 'Invalid image_id'})
      response.status_code = 400
      return response
   else:
      eval_results, cell_counts, image = infer(storage[image_id], upload_dir)
      response = jsonify({'message' : 'Success', 'data': str(eval_results), 'cell_counts': str(cell_counts), 'result_image': request.host + '/' + image})
      response.status_code = 201
      return response

if __name__ == '__main__':
    app.run()