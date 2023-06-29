import firebase_admin
from firebase_admin import credentials, firestore
from entities.Image import Photo


def get_images_from_firestore(user, article):
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(r'/opt/program/briefingaboutit-firebase-adminsdk-192tz-aae98c5f6e.json')
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    user_articles_reference = db.collection(r'Users').document(user.strip()).collection(r'Articles').document(
        article.strip()).get()
    doc_snapshot = user_articles_reference.get('images')
    images = []

    if doc_snapshot is not None:
        if len(doc_snapshot) != 0:
            for doc in doc_snapshot:
                if doc.get('toBlur'):
                    image_id = doc.get('id')
                    image_name = doc.get('imageName')
                    photo = doc.get('photo')
                    image_blurred = doc.get('blurredPhotoAsBitmap')
                    to_blur = doc.get('toBlur')
                    image = Photo(image_id, image_name, photo, image_blurred, to_blur, [])
                    images.append(image)

    return images, doc_snapshot


def commit_images_to_firestore(user, article, images_list):
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(r'/opt/program/briefingaboutit-firebase-adminsdk-192tz-aae98c5f6e.json')
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    user_articles_reference = db.collection(r'Users').document(user.strip()).collection(r'Articles').document(
        article.strip())

    user_articles_reference.set({
        'images': images_list
    }, merge=True)
