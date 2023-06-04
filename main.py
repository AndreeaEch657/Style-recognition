import csv
import hashlib
from flask import Flask, flash, redirect,request
from pandas import read_csv
from ludwig.api import LudwigModel
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json

app = Flask(__name__, '/prediction')

ludwig_model_1 = LudwigModel.load("results\\Impressionism_run\\model")
ludwig_model_2 = LudwigModel.load("results\\Post_Impressionism_run\\model")
ludwig_model_3 = LudwigModel.load("results\\Northern_Renaissance_run\\model")

def replaceCsv(filehash):
    data = []
    with open("uploads\\upload.csv", 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    for row in data:
        row['image_path'] = filehash + '.jpg'

    with open("uploads\\upload.csv", 'w', newline='') as file:
        fieldnames = ['image_path', 'label', 'split']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret' 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return {"response":"No file part"}
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return {"response":"No selected file"}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #filehash = hashlib.md5(open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'rb').read()).hexdigest()
            filehash = hashlib.md5(filename.encode("utf-8")).hexdigest()
            replaceCsv(filehash)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filehash + '.jpg'))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'upload.csv')
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filehash + '.jpg'))
            prediction1, _ = ludwig_model_1.predict(dataset=file_path)
            prediction2, _ = ludwig_model_2.predict(dataset=file_path)
            prediction3, _ = ludwig_model_3.predict(dataset=file_path)
            predictions_array = []
            predictions_array.append(json.loads(prediction1.to_json())['label_probability']['0'])
            predictions_array.append(json.loads(prediction2.to_json())['label_probability']['0'])
            predictions_array.append(json.loads(prediction3.to_json())['label_probability']['0'])
            index = predictions_array.index(max(predictions_array))
            if index == 0:
                return {"response":f'Impressionism{prediction1.to_json()}'}
            if index == 1:
                return {"response":f'Post-Impressionism{prediction2.to_json()}'}
            if index == 2:
                return {"response":f'Northern-Renaissance{prediction3.to_json()}'}

    return '''
    <!doctype html>
    <html>
    <head>
    <link rel="stylesheet" href="prediction\css\styles.css">
    </head>
    <title>Upload new File</title>
    <body>
    <header>
    <h1>Upload new File</h1>
    </header>
    </body>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </html>
    '''




if __name__ == "__main__":
    app.run(debug=True)

