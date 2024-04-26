import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
import pytesseract
from core.config import cfg
import re
import time
from app import VehicleDatabase, db, app, PlateRecognitionLog, ParkingProperties
import matplotlib.pyplot as plt



total_latency = 0  # Initialize total latency
# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# function to recognize license plate numbers using Tesseract OCR
def recognize_plate(img, coords, db_operation_flag):
    # Function body

    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    # grayscale region within bounding box
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu Threshold", thresh)
    #cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 15: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
        try:
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except:
            text = None

        # Keep only the first 7 characters to match the expected license plate format
        plate_num = plate_num[:7]
    if plate_num != "":
        print("License Plate # : ", plate_num)

        if db_operation_flag:
            with app.app_context():
                # Check if the plate exists in the VehicleDatabase
                exists = VehicleDatabase.query.filter_by(num_plate=plate_num).first() is not None
                print(exists)

                if exists:
                    # Query owner of the vehicle
                    query_database = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                    owner_name = query_database.name

                    # Get the vehicle type
                    vehicle_type = VehicleDatabase.query.filter_by(num_plate=plate_num).first().type

                    # Get the latest status of the plate (in or out of the parking)
                    latest_status = PlateRecognitionLog.query.filter_by(plate_num=plate_num).order_by(
                        PlateRecognitionLog.id.desc()).first()
                    status = latest_status.in_out if latest_status else None

                    if vehicle_type == 'car':
                        # Get the current and max slots for cars
                        parking_properties = ParkingProperties.query.first()
                        current_slots = parking_properties.current_slots_car
                        max_slots = parking_properties.max_slots_car

                        if status == 'out' or status is None:
                            # Plate detected and not inside parking
                            if current_slots < max_slots:
                                record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='in')
                                db.session.add(record)

                                # Update VehicleDatabase status to '1' (In)
                                vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                                if vehicle:
                                    vehicle.status = '1'

                                parking_properties.current_slots_car += 1
                                print('The plate is detected and will go in')
                                cv2.waitKey(5000)

                        elif status == 'in':
                            # Plate detected and is inside the parking
                            record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='out')
                            db.session.add(record)

                            # Update VehicleDatabase status to '0' (Out)
                            vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                            if vehicle:
                                vehicle.status = '0'

                            parking_properties.current_slots_car -= 1
                            print('The plate is valid and will go out')
                            cv2.waitKey(5000)
                        else:
                            print('Invalid plate status')


                    elif vehicle_type == 'motor':
                        # Get the current and max slots for cars
                        parking_properties = ParkingProperties.query.first()
                        current_slots = parking_properties.current_slots_car
                        max_slots = parking_properties.max_slots_car

                        if status == 'out' or status is None:
                            # Plate detected and not inside parking
                            if current_slots < max_slots:
                                record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='in')
                                db.session.add(record)

                                # Update VehicleDatabase status to '1' (In)
                                vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                                if vehicle:
                                    vehicle.status = '1'

                                parking_properties.current_slots_car += 1
                                print('The plate is detected and will go in')
                                cv2.waitKey(5000)
                            else:
                                print('Parking is full')

                        elif status == 'in':
                            # Plate detected and is inside the parking
                            record = PlateRecognitionLog(owner_name=owner_name, plate_num=plate_num, in_out='out')
                            db.session.add(record)

                            # Update VehicleDatabase status to '0' (Out)
                            vehicle = VehicleDatabase.query.filter_by(num_plate=plate_num).first()
                            if vehicle:
                                vehicle.status = '0'

                            parking_properties.current_slots_car -= 1
                            print('The plate is valid and will go out')
                            cv2.waitKey(5000)
                        else:
                            print('Invalid plate status')


                    else:
                        print('Invalid vehicle type')

                    db.session.commit()
                else:
                    print('Plate does not exist in the database')

    ##cv2.imshow("Character's Segmented", im2)
    #cv2.waitKey(0)
    return plate_num

def load_freeze_layer(model='yolov3', tiny=False):
    if tiny:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_9', 'conv2d_12']
        else:
            freeze_layouts = ['conv2d_17', 'conv2d_20']
    else:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_58', 'conv2d_66', 'conv2d_74']
        else:
            freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    return freeze_layouts

def load_weights(model, weights_file, model_name='yolov3', is_tiny=False):
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config(FLAGS):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if FLAGS.model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        elif FLAGS.model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes



total_latency = 0.0  # Initialize total latency
frame_count = 0  # Initialize frame count
total_latency = 0.0  # Initialize total latency
frame_count = 0  # Initialize frame count

# Create lists to store latency values
latency_values = []
frame_numbers = []


def draw_bbox(image, bboxes, info=False, counted_classes=None, show_label=True,
              allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values()), read_plate=False,
              db_operation_flag=False):
    global total_latency, frame_count, latency_values, frame_numbers  # Use global variables

    start_time = time.time()

    classes = read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes

    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes:
            continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]

        if class_name not in allowed_classes:
            continue

        else:
            if read_plate:
                height_ratio = int(image_h / 25)
                plate_number = recognize_plate(image, coor, db_operation_flag)

                if plate_number != None:
                    cv2.putText(image, plate_number, (int(coor[0]), int(coor[1] - height_ratio)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 2)

            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if info:
                print(
                    "Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(
                        class_name, score, coor[0], coor[1], coor[2], coor[3]))

            if show_label:
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)
                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

            if counted_classes != None:
                height_ratio = int(image_h / 25)
                offset = 15
                for key, value in counted_classes.items():
                    cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    offset += height_ratio

            end_time = time.time()
            latency = end_time - start_time  # Calculate latency for each frame
            total_latency += latency  # Add latency to total
            frame_count += 1  # Increment frame count

            latency_values.append(latency)
            frame_numbers.append(frame_count)
            ##print("Latency: %.5f seconds" % latency)

        average_latency = total_latency / frame_count if frame_count > 0 else 0

        with open('latency_analysis.txt', 'a') as f:
            f.write(f"{frame_count},{average_latency}\n")

    average_latency = total_latency / frame_count if frame_count > 0 else 0  # Calculate average latency
    #print(f"Average Latency: {average_latency:.5f} seconds")

    return image



def plot_latency_from_file(file_path='latency_analysis.txt'):
    frame_numbers, latency_values = np.loadtxt(file_path, delimiter=',', unpack=True)

    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, latency_values, marker='o', color='b', label='Latency (seconds)')
    plt.axhline(y=np.mean(latency_values), color='r', linestyle='--', label='Average Latency')

    plt.title('Latency Analysis')
    plt.xlabel('Frame Number')
    plt.ylabel('Latency (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # You can also return the average latency if needed
    return np.mean(latency_values)


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
        (
            tf.math.atan(
                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
            )
            - tf.math.atan(
                tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
            )
        )
        * 2
        / np.pi
    ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)

