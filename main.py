import csv
import hashlib
from flask import Flask, flash, request, render_template
#from pandas import read_csv
from ludwig.api import LudwigModel
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json

app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

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
        fieldnames = ['image_path', 'label']
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
            predictions_array.append(json.loads(prediction1.to_json())['label_probabilities_impressionism']['0'])
            predictions_array.append(json.loads(prediction2.to_json())['label_probabilities_post-impressionism']['0'])
            predictions_array.append(json.loads(prediction3.to_json())['label_probabilities_northern-renaissance']['0'])
            index = predictions_array.index(max(predictions_array))
            if index == 0:
                imp_perc = json.loads(prediction1.to_json())['label_probabilities_impressionism']['0']
                postimp_perc = json.loads(prediction1.to_json())['label_probabilities_post-impressionism']['0']
                ren_perc = json.loads(prediction1.to_json())['label_probabilities_northern-renaissance']['0']
                genre = "Impressionism"
                return render_template('index.html',impressionism_percentage=format(imp_perc,'.7f'),postimpressionism_percentage=format(postimp_perc,'.7f'),renaissance_percentage=format(ren_perc,'.7f'),predicted_genre=genre)
            if index == 1:
                imp_perc = json.loads(prediction2.to_json())['label_probabilities_impressionism']['0']
                postimp_perc = json.loads(prediction2.to_json())['label_probabilities_post-impressionism']['0']
                ren_perc = json.loads(prediction2.to_json())['label_probabilities_northern-renaissance']['0']
                genre = "Post-Impressionism"
                return render_template('index.html',impressionism_percentage=format(imp_perc,'.7f'),postimpressionism_percentage=format(postimp_perc,'.7f'),renaissance_percentage=format(ren_perc,'.7f'),predicted_genre=genre)
            if index == 2:
                imp_perc = json.loads(prediction3.to_json())['label_probabilities_impressionism']['0']
                postimp_perc = json.loads(prediction3.to_json())['label_probabilities_post-impressionism']['0']
                ren_perc = json.loads(prediction3.to_json())['label_probabilities_northern-renaissance']['0']
                genre = "Northern-Renaissance"
                return render_template('index.html',impressionism_percentage=format(imp_perc,'.7f'),postimpressionism_percentage=format(postimp_perc,'.7f'),renaissance_percentage=format(ren_perc,'.7f'),predicted_genre=genre)

    return render_template('index.html')




if __name__ == "__main__":
    app.run(debug=True)

