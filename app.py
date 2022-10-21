from keyword_spotting_service import _Keyword_Spotting_Service, Keyword_Spotting_Service
import random
from flask import Flask, render_template,jsonify,request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        return "File has been uploaded."
    return render_template('index.html', form=form)

@app.route('/home', methods=['GET',"POST"])


@app.route("/predict", methods=["POST"])
#  def predict():
# #      """Endpoint to predict keyword
# # :return (json): This endpoint returns a json file with the following format:
# #     {
# #         "keyword": "Kasooli"
# #     }
# #     """
# # get file from POST request and save it
# # file = request.json
# audio_file = request.files['file']
# file_name = str(random.randint(0, 100000))
# audio_file.save(file_name)
# # instantiate keyword spotting service singleton and get prediction
# kss = Keyword_Spotting_Service()
# predicted_keyword = kss.predict(audio_file)
# # we don't need the audio file any more - let's delete it!
# os.remove(file_name)
# # send back result as a json file
# result = {"keyword": predicted_keyword}
# return jsonify('result:', audio_file)
def predict():
    if request.method == 'POST':
        # file = request.json
        # filepath = file['file']
        audio_file = request.files['file']
        file_name = str(random.randint(0, 100000))
        audio_file.save(file_name)
        kss = _Keyword_Spotting_Service()
        predicted_keyword = kss.predict(str(file_name))
        os.remove(file_name)

    return jsonify(predicted_keyword)


if __name__ == "__main__":
    app.run(debug=True)
