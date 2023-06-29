import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

class Photo:
    def __init__(self, id, image_name, photo, image_blurred, to_blur, faces):
        self.id = id
        self.imageName = image_name
        self.photo = photo
        self.imageBlurred = image_blurred
        self.toBlur = to_blur
        self.faces = faces

    def get_faces(self):
        return self.faces

    def is_to_blur(self):
        return self.toBlur

    def get_photo(self):
        return self.photo

    def get_image_blurred(self):
        return self.imageBlurred

    def get_image_name(self):
        return self.imageName

    def get_id(self):
        return self.id

    def get_image_from_bitstring(self):
        bytes = base64.b64decode(self.photo)
        image = Image.open(BytesIO(bytes))
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_cv2

    def set_faces(self, faces):
        self.faces = faces

    def to_dict(self):
        # Convert the Face instances to dictionaries
        faces_dict = [face.__dict__ for face in self.faces]

        # Create a dictionary with the Photo instance properties and the faces dictionary list
        photo_dict = {
            "id": self.id,
            "imageName": self.imageName,
            "photo": self.photo,
            "imageBlurred": self.imageBlurred,
            "toBlur": self.toBlur,
            "faces": faces_dict
        }

        return photo_dict
