import uuid

class Face:
    def __init__(self, face_crop, x_min, y_min, x_max, y_max):
        self.id = str(uuid.uuid4()).replace("/", "-")
        self.faceCrop = face_crop
        self.xMin = x_min
        self.yMin = y_min
        self.xMax = x_max
        self.yMax = y_max

    def get_id(self):
        return self.id

    def get_face_crop(self):
        return self.faceCrop

    def get_x_min(self):
        return self.xMin

    def get_y_min(self):
        return self.yMin

    def get_x_max(self):
        return self.xMax

    def get_y_max(self):
        return self.yMax

