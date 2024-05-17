import os
import threading


# Comment out below line to enable TensorFlow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import requests  # Add this import for making HTTP requests to the Flask server
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov3 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from license_plate_recognizer import recognize_license_plate



flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov3-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov3', 'yolov3')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video2', None, 'path to second input video')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output2', None, 'path to second output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
flags.DEFINE_boolean('campus_count', False, 'display vehicles count on campus')


RECENTLY_RECOGNIZED_PLATES = {}
TIME_WINDOW = 5  # time window in seconds to consider a plate as recently recognized


def get_current_car_count():
    try:
        response = requests.get('http://localhost:5000/current_car_count')
        if response.status_code == 200:
            return response.json().get('current_car_count', 0)
        else:
            return 0
    except requests.RequestException:
        return 0

def process_video(video_path, output_path, infer, input_size, video_name, config, session):
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        print(f'Error opening video file {video_path}')
        return

    out = None
    if output_path:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    frame_num = 0
    last_car_count_update = 0
    current_car_count = get_current_car_count()

    while True:
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed, try a different video format!')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
        image = Image.fromarray(frame)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = list(class_names.values())

        if FLAGS.crop:
            crop_rate = 4  # capture images every so many frames (e.g., crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            os.makedirs(crop_path, exist_ok=True)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path)
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)

                # Check if any objects were cropped before calling recognize_license_plate()
                if len(os.listdir(final_path)) > 0:
                    recognize_license_plate(final_path)
            else:
                pass

        if FLAGS.count:
            counted_classes = count_objects(pred_bbox, by_class=False, allowed_classes=allowed_classes)
            for key, value in counted_classes.items():
                print(f"Number of {key}s: {value}")
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)

        if FLAGS.campus_count and (time.time() - last_car_count_update > 5):
            current_car_count = get_current_car_count()
            last_car_count_update = time.time()

        if FLAGS.campus_count:
            cv2.putText(image, f"Vehicles on Campus: {current_car_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")
        result = np.asarray(image)
        cv2.namedWindow(f"result-{video_name}", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow(f"result-{video_name}", result)

        if output_path:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # Threads for running video processes
    threads = []

    # Process the first video
    video_path = FLAGS.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_path = FLAGS.output if FLAGS.output else None
    thread1 = threading.Thread(target=process_video, args=(video_path, output_path, infer, input_size, video_name, config, session))
    threads.append(thread1)

    # Process the second video if provided
    if FLAGS.video2:
        video_path2 = FLAGS.video2
        video_name2 = video_path2.split('/')[-1].split('.')[0]
        output_path2 = FLAGS.output2 if FLAGS.output2 else None
        thread2 = threading.Thread(target=process_video, args=(video_path2, output_path2, infer, input_size, video_name2, config, session))
        threads.append(thread2)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
