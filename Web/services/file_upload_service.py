from werkzeug.utils import secure_filename
import os
import datetime


class File_Upload_Service:
    ALLOWED_EXTENSIONS = ['jpeg', 'jpg']
    UPLOAD_FOLDER = './static/upload/'

    def __init__(self):
        pass

    @staticmethod
    def move_processed_file(filename):
        source = File_Upload_Service.UPLOAD_FOLDER + filename
        destination = File_Upload_Service.UPLOAD_FOLDER + 'processed/' + str(
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + filename

        os.rename(source, destination)

    def allowed_extension(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def upload_file(self, file):
        if file and self.allowed_extension(file.filename):
            filename = secure_filename(file.filename)
            uploaded_file_path = os.path.join(self.UPLOAD_FOLDER, filename)
            file.save(uploaded_file_path)
            return uploaded_file_path

    def get_file_list(self):

        file_list = []
        for filename in os.listdir(self.UPLOAD_FOLDER):
            path = os.path.join(self.UPLOAD_FOLDER, filename)
            if os.path.isfile(path):
                file_list.append(path)

        return file_list
