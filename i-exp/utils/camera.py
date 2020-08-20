import numpy as np
import torch
from torch.autograd import Variable
import cv2
import dlib
from utils.img_allign_expnet import img_align_modified
from utils.model_phase2_expnet_CPU import ExpNet_p2
# import datetime
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class VideoCamera(object):
    # emo_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    emo_list = ["NEUTRAL", "ANGER", "CONTEMPT", "DISGUST", "FEAR", "HAPPY", "SADNESS", "SURPRISE"]

    def __init__(self):
        self.cv2 = cv2
        self.is_camera_open = True
        self.stop_camera = False
        # self.video = self.cv2.VideoCapture(0)

        # Check if camera opened successfully
        # if not self.video.isOpened():
        #     print("Unable to read camera feed")
        #     self.stop_camera = False

        # self.video_file = os.path.join('./static/videos/',
        #                                str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%s')) + '.avi')

        # self.out_file = os.path.join('./static/captured_images/',
        #                              str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%s')))
        # if not os.path.exists(self.out_file):
        #     os.makedirs(self.out_file)
        #
        # self.out_file += "/pic_{}.jpeg"
        # self.file_ = 0

        # frame_width = int(self.video.get(3))
        # frame_height = int(self.video.get(4))
        # self.video_writer = self.cv2.VideoWriter(self.video_file, self.cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
        #                                          (frame_width, frame_height))

        self.faceCascade = self.cv2.CascadeClassifier('./static/casscade_model/casscade.xml')
        self.model = self.load_model()
        self.dlib_detector, self.dlib_predictor = self.load_dlib()
        self.font = self.cv2.FONT_HERSHEY_SIMPLEX
        self.feelings_faces = []
        # append the list with the emoji images
        for index, emotion in enumerate(self.emo_list):
            self.feelings_faces.append(self.cv2.imread('./static/images/emojis/' + emotion.lower() + '.png', -1))

    def __del__(self):
        # self.is_camera_open = False
        # self.video.release()
        # self.video_writer.release()
        self.cv2.destroyAllWindows()

    def stop(self):
        # self.is_camera_open = False
        # self.video.release()
        # self.video_writer.release()
        self.cv2.destroyAllWindows()

    def load_model(self):
        model_p2 = ExpNet_p2(useCuda=False, gpuDevice=0)
        model_p2.load_state_dict(torch.load(os.path.join('./model', 'expnet_p2.pt'),
                                            map_location=lambda storage, loc: storage))
        # model_p2.load_state_dict(torch.hub.load_state_dict_from_url(os.path.join('./model', 'expnet_p2.pt'),
        #                                     map_location=lambda storage, loc: storage))

        return model_p2

    def load_dlib(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        return detector, predictor

    def get_frame(self, image):
        success, image = True, image  # self.video.read()

        if success:
            model_image, cropped_image = self.format_image(image)
            if model_image is not None:
                result, max_val, max_index = self.predict(model_image)

                if result is not None:
                    # write the different emotions and have a bar to indicate probabilities for each class
                    for index, emotion in enumerate(self.emo_list):
                        self.cv2.putText(image, emotion, (10, index * 20 + 20), self.font, 0.5, (0, 0, 255), 2)
                        self.cv2.rectangle(image, (130, index * 20 + 10),
                                           (130 + int(result[index] * 100), (index + 1) * 20 + 4),
                                           (255, 0, 0), -1)

                    self.cv2.putText(image, self.emo_list[max_index], (10, 360), self.font, 2, (255, 255, 255), 2,
                                     self.cv2.LINE_AA)

                    face_image = self.feelings_faces[max_index]

                    for c in range(0, 3):
                        image[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + image[200:320,
                                                                                                          10:130,
                                                                                                          c] * (
                                                            1.0 - face_image[:, :, 3] / 255.0)

                    gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)

                    faces = self.faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.3,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=self.cv2.CASCADE_SCALE_IMAGE
                    )

                    if len(faces) > 0:
                        # initialize the first face as having maximum area.
                        max_area_face = faces[0]
                        for face in faces:  # (x,y,w,h)
                            if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                                max_area_face = face

                        (x, y, w, h) = max_area_face
                        image = self.cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (255, 255, 255), 2)
                        # self.cv2.imwrite(self.out_file.format(self.file_), image)
                        # self.file_ += 1
                        # self.video_writer.write(image)
                    else:
                        print("No face!!!")
                        self.stop_camera = False
                else:
                    print("No Emotion Prediction!!!.")
                    self.stop_camera = False
            else:
                print("Some issue in image format!!!")
                self.stop_camera = False
        else:
            print("No frame captured from Video!!!")
            self.stop_camera = False

        try:
            return self.cv2.imencode('.jpg', image)[1]  # .tobytes()
        except Exception as e:
            print("Exception while returning image as byte!!!---> {}".format(e))
            self.stop_camera = False
            return bytearray()

    def format_image(self, image):
        try:
            image = img_align_modified(image, self.dlib_detector, self.dlib_predictor)
            original_img = image

            image = self.cv2.cvtColor(image, self.cv2.COLOR_RGB2BGR)
            image = self.cv2.resize(image, (96, 96), interpolation=self.cv2.INTER_LINEAR)
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32) / 255.0
            I = torch.from_numpy(image).unsqueeze(0)
        except Exception as e:
            print("----->Problem during resize:{}".format(e))
            self.stop_camera = True
            return None, None

        return I, original_img

    def predict(self, img):

        test_output = self.model(Variable(img))
        max_val, idx = torch.max(test_output, 1)
        result = test_output.data.cpu().numpy()[0]
        result = np.interp(result, (result.min(), result.max()), (0, +1))
        return result, max_val, idx.data.cpu().numpy()[0]
