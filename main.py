from entities.Image import Photo
from entities.Face import Face


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create a Photo instance with Face instances
    face1 = Face(face_crop="face1", x_min=0, y_min=0, x_max=100, y_max=100)
    face2 = Face(face_crop="face2", x_min=50, y_min=50, x_max=150, y_max=150)
    photo = Photo(id=1, image_name="image.jpg", photo="photo data", image_blurred=True, to_blur=True,
                  faces=[face1, face2])

    # Convert the Photo instance to a dictionary
    photo_dict = photo.to_dict()

    # Print the dictionary
    print(photo_dict)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
