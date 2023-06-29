from collections import Counter
import cv2
import numpy as np


def get_kernels(faces):
    """
    Given a list of face bbox coordinates, returns face kernels
    Input:
    - faces - array of inference results
    Output:
    - kernels - array of face kernels
    """

    kernels = []

    for (x1, y1, x2, y2) in faces:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mean_xy = round(((x2 - x1) + (y2 - y1)) / 2)
        k_size = round(mean_xy * 0.3)
        kernels.append(k_size)

    return kernels


# BLUR FUNCTIONS
def blur_face_ellipse(img, faces):
    """
    Apply ellipse blur with depth feeling to array of faces
    Input:
    - img - image on which to apply blur
    - faces - array of face coordinates (x1, y1, x2, y2)
    Output:
    - img_copy - image with blurred faces
    """

    img_copy = img.copy()
    if faces.size != 0:
        faces = [list(map(int, face)) for face in faces]
        h, w = img_copy.shape[:2]
        kernels = get_kernels(faces)
        kernel_dict = dict(Counter(kernels))
        sorted_kernel_dict = sorted(kernel_dict.items(), key=lambda x: x[0])
        kernels, faces = zip(*sorted(zip(kernels, faces)))

        sorted_kernels, num_kernels = list(zip(*sorted_kernel_dict))
        blurred_img_kernel = [cv2.blur(img_copy, (k_size, k_size)) for k_size in sorted_kernels]

        c_mask = np.zeros((h, w), np.uint8)
        c_masks = []

        current_kernel = kernels[0]

        for i, (x1, y1, x2, y2) in enumerate(faces):
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes_length = ((x2 - x1) // 2, (y2 - y1) // 2)
            if current_kernel != kernels[i]:
                c_masks.append(c_mask)
                c_mask = np.zeros((h, w), np.uint8)
                current_kernel = kernels[i]
            cv2.ellipse(c_mask, center, axes_length, angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
        c_masks.append(c_mask)

        for i in range(len(c_masks)):
            mask = cv2.bitwise_and(img_copy, img_copy, mask=c_masks[i])
            img_mask = img_copy - mask
            mask2 = cv2.bitwise_and(blurred_img_kernel[i], blurred_img_kernel[i], mask=c_masks[i])  # mask
            img_copy = img_mask + mask2

    return img_copy


def blur_face_ellipsekmax(img, faces):
    """
    Apply ellipse blur with max kernel to array of faces
    Input:
    - img - image on which to apply blur
    - faces - array of face coordinates (x1, y1, x2, y2)
    Output:
    - img_copy - image with blurred faces
    """

    img_copy = img.copy()
    if faces.size != 0:
        faces = [list(map(int, face)) for face in faces]
        k_max = 1
        h, w = img_copy.shape[:2]
        c_mask = np.zeros((h, w), np.uint8)

        for (x1, y1, x2, y2) in faces:
            mean_xy = round(((x2 - x1) + (y2 - y1)) / 2)
            k_size = round(mean_xy * 0.3)
            if k_size > k_max:
                k_max = k_size

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes_length = ((x2 - x1) // 2, (y2 - y1) // 2)

            cv2.ellipse(c_mask, center, axes_length, angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)

        mask = cv2.bitwise_and(img_copy, img_copy, mask=c_mask)
        img_mask = img_copy - mask
        blur = cv2.blur(img_copy, (k_max, k_max))
        mask2 = cv2.bitwise_and(blur, blur, mask=c_mask)
        img_copy = img_mask + mask2

    return img_copy


def blur_face_box(img, faces):
    """
    Apply bounding-box blur to array of faces
    Input:
    - img - image on which to apply blur
    - faces - array of face coordinates (x1, y1, x2, y2)
    Output:
    - img_copy - image with blurred faces
    """
    img_copy = img.copy()
    if faces.size != 0:
        faces = [list(map(int, face)) for face in faces]
        img_copy = img[:, :]
        for (x1, y1, x2, y2) in faces:
            mean_xy = round(((x2 - x1) + (y2 - y1)) / 2)
            k_size = round(mean_xy * 0.3)
            roi = img_copy[y1:y2, x1:x2]
            roi = cv2.blur(roi, (k_size, k_size))
            img_copy[y1:y2, x1:x2] = roi

    return img_copy


# blur functions (for form radio)
BLUR_FUNCTIONS = {'ellipse-depth': blur_face_ellipse, 'ellipse-max': blur_face_ellipsekmax, 'box': blur_face_box}
blur_function = None