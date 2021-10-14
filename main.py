# import the necessary packages
# import argparse
import imutils
import time
import cv2
import os
import glob
import math

files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

from sort import *

tracker = Sort()
memory = {}
# 1920x1080p
# line1 = [(380, 900), (1500, 900)]
# line2 = [(710,250), (1160, 250)]
# 1280x720p
line1 = [(250, 600), (1000, 600)]
line2 = [(470, 165), (770, 165)]
counter1 = 0
counter2 = 0

imgName = 0

# construct the argument parse and parse the arguments
input_file_name = 'A03_20200820114721_2'
input_file = 'input_videos\\' + input_file_name + '.mp4'
output_folder = 'output'
output_file = output_folder + '\\' + input_file_name + '_output.mkv'
output_csv_file = output_folder + '\\' + input_file_name + '_output.csv'
yolo = 'yolo-coco'
confidenceLimit = 0.35
threshold = 0.25

########################################################################################################################
import os
import sys
from PIL import Image
import numpy as np


def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(0), "training", "src")
utils_path = os.path.join(get_parent_dir(0), "utils")

sys.path.append(src_path)
sys.path.append(utils_path)

from training.src.keras_yolo3.yolo import YOLO
from timeit import default_timer as timer
from utils.utils import detect_object

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(0), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final_70893.h5")
model_classes = os.path.join(model_folder, "data_classes_70893.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")


########################################################################################################################


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def main_prediction(prediction, imgsize):
    sizefactor = 0.2
    numberofbox = len(prediction)
    if numberofbox == 1:
        temArea = abs(prediction[0][0] - prediction[0][2]) * abs(prediction[0][1] - prediction[0][3])
        # if temArea / float(imgsize) > sizefactor:
        # return input_labels[prediction[0][-2]] + "_{:.4f}".format(prediction[0][-1])
        return input_labels[prediction[0][-2]]
    if numberofbox > 1:
        area = 0
        prebox = []
        for p in prediction:
            # left, top, right, bottom
            temArea = abs(p[0] - p[2]) * abs(p[1] - p[3])
            if area < temArea:
                area = temArea
                prebox = p
            if area == temArea and p[-1] > prebox[-1]:
                prebox = p
        # if area / float(imgsize) > sizefactor:
        # return input_labels[prebox[-2]] + "_{:.4f}".format(prebox[-1])
        return input_labels[prebox[-2]]
    # return None


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo, "yolov3.weights"])
configPath = os.path.sep.join([yolo, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(input_file)
video_fps = vs.get(cv2.CAP_PROP_FPS)
print('Video PFS:', video_fps)
writer = None
# out = None
(W, H) = (None, None)

frameIndex = 0
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": model_classes,
        "score": 0.35,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)

# labels to draw on images
class_file = open(model_classes, "r")
input_labels = [line.rstrip("\n") for line in class_file.readlines()]
# input_labels.sort()
print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

csv_file = open(output_csv_file, "w")
csv_file.write("IndexID,Time,Counter,Label,Speed\n")

id_class_dist = {}
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    detect_img = frame.copy()
    # opencv images are BGR, translate to RGB
    # detect_img = detect_img[:, :, ::-1]

    frame = adjust_gamma(frame, gamma=1.5)
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    center = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidenceLimit:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                center.append(int(centerY))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                # print ('ClassIDs',classIDs)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceLimit, threshold)
    # print("idxs", idxs)
    # print("boxes", boxes[i][0])
    # print("boxes", boxes[i][1])

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])
            # print(confidences[i])
            # print(center[i])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []

    previous = memory.copy()
    # print("centerx",centerX)
    # print("centery",centerY)
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            # # try:
            # factor = 0.5
            # new_y = y - int(abs(h-y)*factor)
            # if new_y < 0:
            #     new_y = 0
            # new_h = h + int(abs(h-y)*factor)
            # if new_h > H:
            #     new_h = H
            # new_x = x - int(abs(w-x)*factor)
            # if new_x < 0:
            #     new_x = 0
            # new_w = w + int(abs(w-x)*factor)
            # if new_w > W:
            #     new_w = W
            #
            # crop_img = detect_img[new_y:new_h, new_x:new_w].copy()
            # # print(y - int(abs(h-y)*factor),h + int(abs(h-y)*factor), x - int(abs(w-x)*factor),w + int(abs(w-x)*factor))
            # # except:
            # #     crop_img = detect_img[y:h, x:w].copy()
            # #     print('Error')
            #
            # # try:
            # # print(crop_img, (x, y), (w, h))
            # crop_img_to_detect = Image.fromarray(crop_img)
            # # if crop_img_to_detect.mode != "RGB":
            # #     crop_img_to_detect = crop_img_to_detect.convert("RGB")
            #
            # scale_percent = 2.0  # percent of original size
            # new_width = int(crop_img_to_detect.width * scale_percent)
            # new_height = int(crop_img_to_detect.height * scale_percent)
            # dim = (new_width, new_height)
            # # resize image
            # resized_image = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
            # # cv2.imshow("resized", resized_image)
            #
            # crop_img_to_detect = Image.fromarray(resized_image)
            # if crop_img_to_detect.mode != "RGB":
            #     crop_img_to_detect = crop_img_to_detect.convert("RGB")
            # prediction, new_image = yolo.detect_image(crop_img_to_detect, show_stats=False)
            # print('Prediction:', indexIDs[i], prediction)
            #
            # # vehicle_label = 'None_0'
            # if len(prediction) > 0:
            #     # vehicle_label = input_labels[prediction[0][-2]] + "_{:.4f}".format(prediction[0][-1])
            #     vehicle_label = main_prediction(prediction, crop_img_to_detect.width*crop_img_to_detect.height)
            #     if str(indexIDs[i]) in id_class_dist:
            #         id_class_dist[str(indexIDs[i])].append(vehicle_label)
            #     else:
            #         id_class_dist[str(indexIDs[i])] = [vehicle_label]
            #     # if vehicle_label is not None:
            #         # cv2.imwrite(output_folder + '\\' + str(indexIDs[i]) + '_' + vehicle_label + '_' + str(imgName) + '.jpg', resized_image)
            #         # new_image.save(os.path.join(output_folder, str(indexIDs[i]) + '_' + vehicle_label + '_' + str(imgName) + '.jpg'))
            # # except Exception as e:
            # #     print(e)
            # imgName += 1

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                cv2.line(frame, p0, p1, color, 3)

                # Speed Calculation
                y_pix_dist = int(y + (h - y) / 2) - int(y2 + (h2 - y2) / 2)
                text_y = "{} y".format(y_pix_dist)
                x_pix_dist = int(x + (w - x) / 2) - int(x2 + (w2 - x2) / 2)
                text_x = "{} x".format(x_pix_dist)
                # cv2.putText(frame, text_y, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                # cv2.putText(frame, text_x, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                final_pix_dist = math.sqrt((y_pix_dist * y_pix_dist) + (x_pix_dist * x_pix_dist))
                speed = np.round(2.0 * y_pix_dist, 2)
                text_speed = "ID: {}, Speed: {} km/h".format(indexIDs[i], abs(speed))
                cv2.putText(frame, text_speed, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if intersect(p0, p1, line1[0], line1[1]):
                    counter1 += 1
                    # pre_class = id_class_dist[str(indexIDs[i])]
                    # class_label = max(pre_class,key=pre_class.count)
                    try:
                        video_fps = 15
                        current_time_sec = frameIndex / float(video_fps)
                        current_time_clock = '{0:02.0f}:{1:02.2f}'.format(*divmod(current_time_sec, 60))
                        print("Time:", current_time_clock, "Counter:", counter1, '\tIndexID:', indexIDs[i],
                              '\tLable', LABELS[classIDs[i]], '\tSpeed', speed)
                        csv_file.write(str(indexIDs[i]) + ',' + current_time_clock + ',' + str(counter1) + ','
                                       + LABELS[classIDs[i]] + ',' + str(speed) + "\n")
                        # text = "{}".format(class_label)
                        # cv2.putText(frame, text, (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # crop_img = detect_img[y - 30:h + 30, x - 30:w + 30].copy()
                        # cv2.imshow("cropped", detect_img)
                        # cv2.imwrite('output\\' + str(imgName) + '.jpg', crop_img)

                        # detect_img = Image.fromarray(detect_img)
                        # # if detect_img.mode != "RGB":
                        # #     detect_img = detect_img.convert("RGB")
                        #
                        # prediction, new_image = yolo.detect_image(detect_img)
                        # print('Prediction:', prediction)
                        # cv2.imwrite('output\\new_image_' + str(imgName) + '.jpg', new_image)

                        # imgName += 1
                    except Exception as e:
                        print(e)
            #                if intersect(p0, p1, line2[0], line2[1]):
            #                    counter2 += 1

            # text = "{}, {}, {:.4f}".format(indexIDs[i], LABELS[classIDs[i]], confidences[i])
            # print(text)
            # text = "{}".format(indexIDs[i])
            # cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # draw line
    cv2.line(frame, line1[0], line1[1], (0, 255, 255), 4)
    cv2.line(frame, line2[0], line2[1], (0, 255, 255), 4)

    note_text = "NOTE: Vehicle speeds are calibrated only at yellow line. speed of cars are more stable."
    # cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    # draw counter
    counter_text = "Counter:{}".format(counter1)
    cv2.putText(frame, counter_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)
    #    cv2.putText(frame, "ctr2",str(counter2), (100,400), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
    # counter += 1

    # saves image file
    # +cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

    # cv2.imshow('result', frame)
    # out = cv2.VideoWriter('output\\CCTV2-output.mkv',
    #                       cv2.VideoWriter_fourcc(*"MJPG"), 15, (1920, 1080), True)
    # out.write(frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_file, fourcc, 15, (1280, 720), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

    # if frameIndex >= 4000: # limits the execution to the first 4000 frames
    #    print("[INFO] cleaning up...")
    #    writer.release()
    #    vs.release()
    #    exit()

    if cv2.waitKey(33) == 27:
        break

yolo.close_session()
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
csv_file.close()
cv2.destroyAllWindows()
