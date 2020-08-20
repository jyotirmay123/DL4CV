import cv2
import dlib

from utils.utills import extract_left_eye_center, extract_right_eye_center, angle_between_2_points, crop_image

emo_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def img_align(pathname, label=None):
    scale = 1
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

    img = cv2.imread(pathname)
    height, width = img.shape[:2]
    s_height, s_width = height // scale, width // scale
    img = cv2.resize(img, (s_width, s_height))

    dets = detector(img, 1)

    for i, det in enumerate(dets):
        shape = predictor(img, det)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        cropped = crop_image(rotated, det)
        if label != None:
            image_name = pathname.split("/")[-1]
            cv2.imwrite("data/{0}/{1}".format(emo_list[label], image_name), cropped)

        return cropped


def img_align_modified(img, detector, predictor):
    scale = 1
    height, width = img.shape[:2]
    s_height, s_width = height // scale, width // scale
    img = cv2.resize(img, (s_width, s_height))

    dets = detector(img, 1)

    for i, det in enumerate(dets):
        shape = predictor(img, det)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        cropped = crop_image(rotated, det)

        return cropped
