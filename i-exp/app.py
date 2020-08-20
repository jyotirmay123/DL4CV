# from PIL import Image
# import imutils
from flask import Flask, request, Response, send_file, send_from_directory, render_template, url_for, jsonify
from utils.utills import image_base64, bytes_to_base64
import numpy as np
from utils.camera import VideoCamera
import io
import cv2

app = Flask(__name__)
vc = VideoCamera()


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')  # Put any other methods you need here
    return response


@app.route('/')
def hello_world():
    return render_template('video.html')


@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image
        # finally run the image through tensor flow object detection`
        # frame = np.frombuffer(image_file)
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        frame = cv2.imdecode(data, color_image_flag)
        # image_object = Image.open(image_file)
        # frame = imutils.resize(image_object, width=700)
        # print(frame.shape)

        bytes_res = vc.get_frame(frame)

        # bytes_res = cv2.imencode('.jpg', frame)[1]

        return jsonify({'feed': f"data:image/jpeg;base64,{bytes_to_base64(bytes_res)}"})
    except Exception as e:
        print('POST /image error: %e' % e)
        return e


if __name__ == '__main__':
    app.run()
