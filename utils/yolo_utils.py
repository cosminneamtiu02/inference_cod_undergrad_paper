import logging
import math
import time
import onnxruntime as ort
import base64

from utils.blur_funcs import *

MAX_SIZE = 640



# models
MODEL_NAMES = {'tiny': 'yolov5n-0.5-face', 'small': 'yolov5s-face', 'medium': 'yolov5m-face', 'large': 'yolov5l-face'}

logger = logging.getLogger("nofacebook")


# MAIN YOLOFACE FUNCTIONS
def yoloface_preprocessing(orig_x, s3_bucket=None, is_path=True):
    """
    Apply preprocessing to image.
    Input:
    - orig_x - image (or path to image if 'is_path' is set to True)
    - is_path - whether image is actually path to image
    - s3_bucket - S3 bucket object (for retrieving file)
    Output:
    - img - final, preprocessed image
    """
    if is_path:
        if s3_bucket:
            orig_x = s3_bucket.Object(orig_x).get().get("Body").read()
            orig_x = cv2.imdecode(np.asarray(bytearray(orig_x)), cv2.IMREAD_COLOR)
        else:
            orig_x = cv2.imread(orig_x)

    preproc_x, no_pad_shape = resize_img(orig_x, max_size=MAX_SIZE,
                                         preserve_ratio=True, switch_channels=True)

    # normalization
    preproc_x = preproc_x.astype(np.float)
    preproc_x /= 255.0

    # add batch size to shape
    if preproc_x.ndim == 3:
        preproc_x = np.expand_dims(preproc_x, axis=0)

    return orig_x, preproc_x.astype(np.float32), no_pad_shape


def yoloface_inference(x, model_type=None, model_path=None, webcam=False):
    """
    Detect faces in one image using a YOLOv5 ONNX inference session.
    Input:
    - x - preprocessed image
    - model_type - type of YOLO5Face model
    Output:
    - bbox_coords - bounding box coordinates
    """
    if webcam:
        return None

    else:
        if model_type is not None:
            model = MODEL_NAMES.get(model_type)
            infer_sess = ort.InferenceSession(f'models/{model}.onnx')

        else:
            infer_sess = ort.InferenceSession(model_path)

        # pass image to inference session, get outputs
        return infer_sess.run(None, {'input': x})


def yoloface_postprocessing(orig_x, no_pad_shape, outputs, blur_name,
                            conf_threshold=0.3, iou_threshold=0.5):
    # get detections from outputs
    detections = outputs[0]

    # apply non-max suppression face
    # (preds that exceed IoU and confidence threshold)
    detections = non_max_suppression_face(detections, conf_threshold, iou_threshold)[0]

    # extract and rescale coordinates
    bbox_coords = detections[:, :4].copy()
    faces = rescale_coordinates(old_shape=no_pad_shape[:2],
                                new_shape=orig_x.shape[:2],
                                coords=bbox_coords,
                                pad=get_padding(no_pad_shape))

    blur_f = BLUR_FUNCTIONS.get(blur_name)
    # apply blur on image and return blurred image
    blurred_img = blur_f(img=orig_x, faces=faces)
    return blurred_img


# UTILS FOR YOLOFACE FUNCTIONS

def resize_img(img, max_size=MAX_SIZE, preserve_ratio=True,
               switch_channels=False, padding=True,
               verbose=False, is_path=False):
    """
    Resizes image for YOLO5Face.
    Input:
    - img - image (if 'is_path' is true, this is actually the path to the image, else it is a webcam stream photo)
    - max_size - length of longest side
    - preserve_ratio - whether to preserve ratio between longer and shorter side
      (False might lead to stretched / squished data, padding is recommended)
    - switch_channels - switch BGR to RGB
    - padding - whether to resize with padding (recommended)
    - is_path - whether 'img' is actually the path to img
    Output:
    - new_img - resized image
    - no_pad_shape - shape without padding (for coordinate rescaling)
    """

    # read image, extract shape
    if is_path:
        img = cv2.imread(img)

    h, w = img.shape[:2]

    # calculate new dimensions
    if not preserve_ratio:
        # dimensions without aspect ratio (not recommended)
        new_dims = (max_size, max_size)
    else:
        # dimensions with aspect ratio preserved (recommended)
        resize_factor = max_size / max(h, w)

        if verbose:
            logger.info(f'resize factor:{resize_factor}')

        new_dims = (int(w * resize_factor), int(h * resize_factor))

        if verbose:
            logger.info(f'new dims: {new_dims}')

    if verbose:
        logger.info(f'old dims: {w}x{h}\nnew dims: {new_dims[0]}x{new_dims[1]}')

    # resize image
    new_img = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)

    # save old shape (for rescaling later)
    no_pad_shape = new_img.shape

    # apply padding
    if padding:
        new_h, new_w = new_img.shape[:2]
        pad_h = math.ceil((max_size - new_h) / 2)
        pad_w = math.ceil((max_size - new_w) / 2)

        new_img = cv2.copyMakeBorder(new_img, pad_h, pad_h,
                                     pad_w, pad_w, cv2.BORDER_CONSTANT)
        new_img = new_img[0:max_size, 0:max_size]

    # BGR to RGB, channels first
    if switch_channels:
        new_img = new_img[:, :, ::-1].transpose(2, 0, 1).copy()

    return new_img, no_pad_shape


def readb64(some_uri):
    """
    Converts uri received from frontend to cv2 image
    Input:
    - uri
    Output:
    - some_img - cv2 image
    """

    encoded_data = some_uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    some_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return some_img


def frame_to_bytes(frame):
    """
    Converts cv2 image to bytes
    Input:
    - frame
    Output:
    - frame converted to bytes
    """
    success, encoded_image = cv2.imencode('.png', frame)
    return encoded_image.tobytes()


def xywh2xyxy(x):
    """
    Converts nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2].
    xy1 = top-left, xy2 = bottom-right
    Input:
    - x - boxes that need to be converted
    Output:
    - y - converted boxes
    """

    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def rescale_coordinates(old_shape, new_shape, coords, pad=None, verbose=False):
    """
    Rescale (bounding box) coordinates from resized to original image.
    Input:
    - old_shape - shape of resized image
    - new_shape - shape of original image
    - coords - coordinate tensor (4 points for each coordinate)
    - pad - padding applied upon resizing (optional)
    Output:
    - coords - rescaled coordinates
    """

    if verbose:
        logger.info(f'scaling from {old_shape} to {new_shape}')

    h_old, w_old = old_shape
    h_new, w_new = new_shape

    x_ratio = w_new / w_old
    y_ratio = h_new / h_old

    pad_left = pad[0]
    pad_bottom = pad[1]

    if pad_left % 2 == 1:
        pad_left += 1

    pad_left //= 2
    pad_bottom //= 2

    if verbose:
        logger.info(f'x_ratio: {w_new} / {w_old} = {x_ratio}')
        logger.info(f'y_ratio: {h_new} / {h_old} = {y_ratio}')

    # iterate on coordinates
    for i, coord in enumerate(coords):
        # get coordinate points
        x1, y1, x2, y2 = coord

        # calculate new coordinate points
        new_x1, new_x2 = tuple([((x - pad_left) * x_ratio) for x in (x1, x2)])
        new_y1, new_y2 = tuple([((y - pad_bottom) * y_ratio) for y in (y1, y2)])

        # Check if coordinates are inside the image
        if new_x1 < 0: new_x1 = 0
        if new_y1 < 0: new_y1 = 0
        if new_x2 > w_new: new_x2 = w_new
        if new_y2 > h_new: new_y2 = h_new

        # add new coordinate points to tensor
        coords[i] = np.array([new_x1, new_y1, new_x2, new_y2]).copy()

    return coords


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Input:
    - boxes1 (Array[N, 4]), boxes2 (Array[M, 4]) - sets of boxes
    Output:
    - iou (Array[N, M]) - the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    This implementation is taken from the above link and changed so that it only uses numpy.
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def nms(boxes, scores, thresh):
    """
    Given boxes and confidence scores, keep only boxes whose score exceeds a given threshold
    Input:
    - boxes - numpy array of boxes (x1, y1, x2, y2)
    - scores - box confidence scores
    - thresh - threshold
    Output:
    - keep - boxes whose score exceeds 'thresh'
    """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more iou's first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maximum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maximum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def non_max_suppression_face(prediction, conf_thresh=0.25, iou_thresh=0.45,
                             classes=None, agnostic=False, labels=()):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Input:
    - prediction - array of inference predictions
    - conf_thresh - confidence threshold
    - iou_thresh - intersection over union threshold
    Output:
    - output - detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15
    xc = prediction[..., 4] > conf_thresh  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 16))] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lab = labels[xi]
            v = np.zeros((len(lab), nc + 15))
            v[:, :4] = lab[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lab)), lab[:, 0].long() + 15] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 15:] > conf_thresh).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 15, None], x[i, 5:15], j[:, None].float()), 1)
        else:  # best class only
            conf = x[:, 15:].max(axis=1, keepdims=True)
            j = x[:, 15:].argmax(axis=1, keepdims=True)
            x = np.concatenate((box, conf, x[:, 5:15], j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thresh]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = nms(boxes, scores, iou_thresh)

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thresh  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.matmul(weights, x[:, :4]).astype(np.double) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def get_padding(no_pad_shape):
    pad_x = (MAX_SIZE - no_pad_shape[1])
    pad_y = (MAX_SIZE - no_pad_shape[0])

    return pad_x, pad_y


"""
def evaluate_widerface(folder_pict_path, widerface_folder, save_folder,
                       infer_sess, write_images=False):
    '''
    Evaluate ONNX inference session on WIDERFACE validation set.
    Input:
    - folder_pict_path - a txt file that lists each folder and the photos it contains
    - widerface_folder - path to the widerface val images (all in one folder)
    - save_folder - path in which to save results
    - infer_sess - ONNX inference session
    - write_images - whether to save processed images
    '''
    pict_folder = {}
    with open(folder_pict_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\\', '/').strip().split('/')
            pict_folder[line[-1]] = line[-2]
    for image_path in glob.glob(os.path.join(widerface_folder, '*')):
        if image_path.endswith('.txt'):
            continue
        logger.info(f'processing {image_path}')
        '''
        final_image, (bboxes, conf) = detect_image(image_path, infer_sess,
                                                   conf_threshold=0.02, iou_threshold=0.5,
                                                   for_evaluation=True, time_code=False,
                                                   postprocessing_f=blur_face_ellipse, show_boxes=False)
        '''
        # path for saving result
        image_name = os.path.basename(image_path)
        txt_name = os.path.splitext(image_name)[0] + ".txt"
        save_name = os.path.join(save_folder, pict_folder.get(image_name, ''), txt_name)
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if write_images:
            cv2.imwrite(f'{dirname}/{image_name}', final_image)
        # open result file, write results
        with open(save_name, 'w') as fd:
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxes_num = str(len(bboxes)) + "\n"
            fd.write(file_name)
            fd.write(bboxes_num)
            for box, conf in zip(bboxes, conf):
                x1, y1, x2, y2 = box
                fd.write("%d %d %d %d %.03f" %
                         (x1, y1, x2 - x1, y2 - y1,
                          min(conf, 1)) + "\n")
"""
