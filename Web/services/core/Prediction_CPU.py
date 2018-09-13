import numpy as np
import torch
from torch.autograd import Variable
import cv2
import services.core.utils.img_allign_expnet as iae
import services.core.utils.model_phase2_expnet_CPU as mpe
import services.file_upload_service as file_upload_service
import os

emo_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def ReadImage(pathname, label=None, isAlligned=True):
    if isAlligned:
        img = cv2.imread(pathname)
    else:
        img = iae.img_align(pathname, label)
    original_img = img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    I_ = torch.from_numpy(img).unsqueeze(0)

    return I_, original_img


def Execute():
    model_p2 = mpe.ExpNet_p2(useCuda=False, gpuDevice=0)
    model_p2.load_state_dict(torch.load(os.path.join('./services/core/model', 'expnet_p2.pt'), \
                                        map_location=lambda storage, loc: storage))

    folder = "static/upload"
    res = []
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            img, cropped = ReadImage(file_path, isAlligned=False)
            test_output = model_p2(Variable(img, volatile=True))
            max_val, idx = torch.max(test_output, 1)
            res.append(dict(image_path=file_path, emotion=emo_list[idx.data.cpu().numpy()[0]]))
            file_upload_service.File_Upload_Service.move_processed_file(the_file)

    return res
