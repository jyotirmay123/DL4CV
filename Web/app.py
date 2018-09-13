import services.core.Prediction_CPU as prediction
import services.file_upload_service as file_upload_service
from flask import Flask, send_file, render_template, redirect, flash, request, url_for, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', data='Bloomberg Magic!')


@app.route('/upload_files', methods=['POST'])
def upload_files():
    file = request.files
    file_part = file['data_file']

    file_upload_service_obj = file_upload_service.File_Upload_Service()
    file_upload_service_obj.upload_file(file=file_part)

    return redirect(url_for('index'))


@app.route('/file_list', methods=['GET'])
def file_list():
    file_upload_service_obj = file_upload_service.File_Upload_Service()
    files = file_upload_service_obj.get_file_list()
    return jsonify(files) if len(files) is not 0 else jsonify(["No File Found!"])


@app.route('/fetch_emotion', methods=['POST', 'GET'])
def process():
    res = prediction.Execute()
    return jsonify(res) if len(res) is not 0 else jsonify(dict(err="No File to Process!!"))


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
