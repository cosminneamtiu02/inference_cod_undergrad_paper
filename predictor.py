import glob
import logging
import os
import json
import cv2
import base64

from flask import request
import flask

from utils.app_config import AppConfig
from utils.yolo_utils import yoloface_inference, yoloface_preprocessing, non_max_suppression_face, rescale_coordinates, \
    get_padding
from utils.firebase_utils import get_images_from_firestore, commit_images_to_firestore
from entities.Face import Face

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

logger = logging.getLogger()
model_path = "/opt/ml/models"

app = flask.Flask(__name__)
app.secret_key = "secret key"
local = False
app_config = AppConfig('config.yaml', local)


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded
    model_name = None

    @classmethod
    def get_model(cls):
        logger.info(f"/opt/ml: {os.listdir('/opt/ml/')}")
        logger.info(f"/glob opt/ml: {glob.glob('/opt/ml/**', recursive=True)}")

        return r'/opt/ml/model/yolov5m-face.onnx'

    @classmethod
    def predict(cls, image, model):
        """For the input, do the predictions and return them.
        Args:
            image : The image on which to do the predictions(cv2 image).
            model : The loaded model used for the predictions.

        """

        orig_x, x_preprocessed, no_pad_shape = yoloface_preprocessing(image, is_path=False)

        outputs = yoloface_inference(x=x_preprocessed, model_path=model, model_type=None)

        detections = non_max_suppression_face(prediction=outputs[0], conf_thresh=0.3, iou_thresh=0.5)[0]

        # extract and rescale coordinates
        bbox_coords = detections[:, :4].copy()
        faces_bbox = rescale_coordinates(old_shape=no_pad_shape[:2],
                                         new_shape=orig_x.shape[:2],
                                         coords=bbox_coords,
                                         pad=get_padding(no_pad_shape))

        return faces_bbox


@app.route("/ping", methods=["GET"])
def ping():
    return "Ok"


@app.route("/models", methods=["GET"])
def get_models():
    models = os.listdir("models")
    return models


@app.route('/invocations', methods=['POST'])
def upload_image():
    request_data = request.get_json()
    request_as_json = json.loads(request_data.replace("'", '"'))

    user = request_as_json["user"]
    article_id = request_as_json["article_id"]

    logging.info(f"recieved: {user} ")
    logging.info(f"recieved: {article_id} ")


    if user is None or not isinstance(user, str):
        return flask.Response("Invalid input_key", 400)

    if article_id is None or not isinstance(article_id, str):
        return flask.Response("Invalid input_key", 400)

    images, response = get_images_from_firestore(user, article_id)

    logging.info(f"database status: {response}")

    if len(images) != 0:
        for image in images:
            single_image = image.get_image_from_bitstring()
            bboxes = ScoringService.predict(image=single_image,
                                            model=ScoringService.get_model())

            logging.info(f"Inference successful and returning {len(bboxes)} faces...")

            faces = []

            for bbox in bboxes:
                x_min = int(bbox[0])
                logging.info(f"x_min: {x_min}")
                y_min = int(bbox[1])
                logging.info(f"y_min: {y_min}")
                x_max = int(bbox[2])
                logging.info(f"x_max: {x_max}")
                y_max = int(bbox[3])
                logging.info(f"y_max: {y_max}")
                face_crop = single_image[y_min:y_max, x_min:x_max]

                retval, crop_as_bitmap = cv2.imencode('.bmp', face_crop)

                # Convert bitmap to base64 string
                crop_as_bitmap_to_string = base64.b64encode(crop_as_bitmap).decode('utf-8')

                faces.append(Face(crop_as_bitmap_to_string, x_min, y_min, x_max, y_max))

            image.set_faces(faces)

    commit_images_to_firestore(user, article_id, [image.to_dict() for image in images])

    return flask.Response("Inference successful", 200)


def get_path(directory, file):
    # Walk through the directory tree and look for the file
    for root, dirs, files in os.walk(directory):
        if file in files:
            # If the file is found, return its absolute path
            file_path = os.path.abspath(os.path.join(root, file))
            return f'The file "{file}" was found at "{file_path}"'

    else:
        # If the file is not found, print an error message
        return f'Error: Could not find the file "{file}" in directory "{directory}"'