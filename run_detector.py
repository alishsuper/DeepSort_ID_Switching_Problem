import numpy as np
import pandas as pd
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import math
import warnings
import csv

import mylib.io as myio
from mylib.displays import drawActionResult
import mylib.funcs as myfunc
import mylib.feature_proc as myproc 
from mylib.action_classifier import ClassifierOnlineTest
from mylib.action_classifier import *

from dtw.dtw import dtw

CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
DRAW_FPS = True

################### DeepSort+YOLO
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')
###################
from utils import choose_run_mode, set_video_writer
# INPUTS ==============================================================

def parse_input_method():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=False, default='webcam', 
                        choices=["webcam", "folder", "txtscript"])
    return parser.parse_args().source
 
arg_input = parse_input_method()
FROM_FOLDER = arg_input == "folder" # from web camera

# PATHS and SETTINGS =================================#############################

'''
# function to check skeletons
def dif(a, b):
    dif = 0
    for k in range(len(a)):
        for j in range(len(b)):
            for i in range(18):
                dif = abs(a[k][i]-b[j][i])
                if dif == 0 and k != j:
                    b[j] = a[j]
                    b[k] = a[k]
                    print("yes", k , j)
                    break
    return b
'''
# Choose image_size from: ["640x480", "432x368", "304x240", "240x208", "208x160"]
# The higher, the better. But slower.
'''
def scan_smartwatch():
# first person
    date_p = []
    time_p = []
    walk_p = []
    jump_p = []
    stand_p = []
    falldown_p = []
    b1 = []
    with open('src/smartwatch/process_A.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            date_p.append(row[0])
            time_p.append(row[1])
            walk_p.append(row[2])
            jump_p.append(row[3])
            stand_p.append(row[4])
            falldown_p.append(row[5])
    date_p.pop(0)
    time_p.pop(0)
    walk_p.pop(0)
    jump_p.pop(0)
    stand_p.pop(0)
    falldown_p.pop(0)
    for i in range(len(date_p)):
        if walk_p[i] == '1':
            b1.append(1)
        if jump_p[i] == '1':
            b1.append(2)
        if stand_p[i] == '1':
            b1.append(3)
        if falldown_p[i] == '1':
            b1.append(4)
# second person
    date_p = []
    time_p = []
    walk_p = []
    jump_p = []
    stand_p = []
    falldown_p = []
    b2 = []
    with open('src/smartwatch/process_B.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            date_p.append(row[0])
            time_p.append(row[1])
            walk_p.append(row[2])
            jump_p.append(row[3])
            stand_p.append(row[4])
            falldown_p.append(row[5])
    date_p.pop(0)
    time_p.pop(0)
    walk_p.pop(0)
    jump_p.pop(0)
    stand_p.pop(0)
    falldown_p.pop(0)
    for i in range(len(date_p)):
        if walk_p[i] == '1':
            b2.append(1)
        if jump_p[i] == '1':
            b2.append(2)
        if stand_p[i] == '1':
            b2.append(3)
        if falldown_p[i] == '1':
            b2.append(4)
    for i in range(27):
        b2.pop(0)
    return b1, b2
'''
'''
def scan_video():
    file1 = open("src/smartwatch/actions.txt","r")
    lines = file1.read().split('\n')
    a1 = []
    a2 = []
    a3 = []
    k = 0
    for i in range(0, len(lines)):
        if k == 0:
            if lines[i] == 'Walk':
                a1.append(1)
            if lines[i] == 'Jump':
                a1.append(2)
            if lines[i] == 'Stand':
                a1.append(3)
            if lines[i] == 'FallDown':
                a1.append(4)
        if k == 1:
            if lines[i] == 'Walk':
                a2.append(1)
            if lines[i] == 'Jump':
                a2.append(2)
            if lines[i] == 'Stand':
                a2.append(3)
            if lines[i] == 'FallDown':
                a2.append(4)
        if k == 2:
            if lines[i] == 'Walk':
                a3.append(1)
            if lines[i] == 'Jump':
                a3.append(2)
            if lines[i] == 'Stand':
                a3.append(3)
            if lines[i] == 'FallDown':
                a3.append(4)
        k += 1
        if k == 3:
            k = 0
    return a1, a2, a3
'''
'''
def calculate_dtw(a1, a2, a3, b1, b2):

    # a1 b1
    x = np.asarray(a1)#np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    y = np.asarray(b1)#np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    print(x, y)

    euclidean_norm = lambda x, y: np.abs(x - y)

    d0, _, _, _ = dtw(x, y, dist=euclidean_norm)

    print(d0)

    # a2 b1
    x = np.asarray(a2)
    y = np.asarray(b1)

    euclidean_norm = lambda x, y: np.abs(x - y)

    d1, _, _, _ = dtw(x, y, dist=euclidean_norm)

    print(d1)

    # a3 b1
    x = np.asarray(a3)
    y = np.asarray(b1)

    euclidean_norm = lambda x, y: np.abs(x - y)

    d2, _, _, _ = dtw(x, y, dist=euclidean_norm)

    print(d2)

    # a1 b2
    x = np.asarray(a1)#np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    y = np.asarray(b2)#np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

    euclidean_norm = lambda x, y: np.abs(x - y)

    d3, _, _, _ = dtw(x, y, dist=euclidean_norm)

    print(d3)

    # a2 b2
    x = np.asarray(a2)
    y = np.asarray(b2)

    euclidean_norm = lambda x, y: np.abs(x - y)

    d4, _, _, _ = dtw(x, y, dist=euclidean_norm)

    print(d4)

    # a3 b2
    x = np.asarray(a3)
    y = np.asarray(b2)

    euclidean_norm = lambda x, y: np.abs(x - y)

    d5, _, _, _ = dtw(x, y, dist=euclidean_norm)

    print(d5)
    return d0, d1, d2, d3, d4, d5
'''
#######################################################################################

if FROM_FOLDER:
    folder_suffix = "4"
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    def set_source_images_from_folder():
        return CURR_PATH + "../data_test/DJI_0140/", 1
    SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES = set_source_images_from_folder()
    folder_suffix += SRC_IMAGE_FOLDER.split('/')[-2] # plus folder name
    # image_size = "304x240"
    image_size = "432x368"
    #image_size = "720x640"#"640x480"#"1600x1200"
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]
    LOAD_MODEL_PATH = CURR_PATH + "../model/trained_classifier_4actions.pickle"
    action_labels=  ['FallDown', 'Jump', 'Stand', 'Walk']

else:
    assert False

if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:
    SKELETON_FOLDER = CURR_PATH + "skeleton_data/"
    SAVE_DETECTED_SKELETON_TO =         CURR_PATH + "skeleton_data/skeletons"+folder_suffix+"/"
    SAVE_DETECTED_SKELETON_IMAGES_TO =  CURR_PATH + "skeleton_data/skeletons"+folder_suffix+"_images/"
    SAVE_IMAGES_INFO_TO =               CURR_PATH + "skeleton_data/images_info"+folder_suffix+".txt"

    # create folders for saving results
    if not os.path.exists(SKELETON_FOLDER):
        os.makedirs(SKELETON_FOLDER)
    if not os.path.exists(SAVE_DETECTED_SKELETON_TO):
        os.makedirs(SAVE_DETECTED_SKELETON_TO)
    if not os.path.exists(SAVE_DETECTED_SKELETON_IMAGES_TO):
        os.makedirs(SAVE_DETECTED_SKELETON_IMAGES_TO)

# Openpose include files and configs ==============================================================

sys.path.append(CURR_PATH + "githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---- For tf 1.13.1, The following setting is needed
import tensorflow as tf
from tensorflow import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# If GPU memory is small, modify the MAX_FRACTION_OF_GPU_TO_USE
MAX_FRACTION_OF_GPU_TO_USE = 0.5
config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU_TO_USE

# Openpose Human pose detection ==============================================================

class SkeletonDetector(object):
    # This func is mostly copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model=None, image_size=None):
        
        if model is None:
            model = "cmu"

        if image_size is None:
            image_size = "432x368" # 7 fps
            # image_size = "336x288"
            # image_size = "304x240" # 14 fps

        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        self.resize_out_ratio = 4.0

        w, h = model_wh(image_size)
        if w == 0 or h == 0:
            e = TfPoseEstimator(
                    get_graph_path(self.model),
                    target_size=(432, 368),
                    tf_config=config)
        else:
            e = TfPoseEstimator(
                get_graph_path(self.model), 
                target_size=(w, h),
                tf_config=config)

        # self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()
        self.cnt_image = 0

    def detect(self, image):
        self.cnt_image += 1
        if self.cnt_image == 1:
            self.image_h = image.shape[0]
            self.image_w = image.shape[1]
            self.scale_y = 1.0 * self.image_h / self.image_w
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)

        # Print result and time cost
        #elapsed = time.time() - t
        #logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        # logger.debug('show+')
        if DRAW_FPS:
            cv2.putText(img_disp,
                        # "Processing speed: {:.1f} fps".format( (1.0 / (time.time() - self.fps_time) )),
                        "fps = {:.1f}".format( (1.0 / (time.time() - self.fps_time) )),
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        self.fps_time = time.time()

    def humans_to_skelsList(self, humans, scale_y = None): # get (x, y * scale_y)
        # type: humans: returned from self.detect()
        # rtype: list[list[]]
        if scale_y is None:
            scale_y = self.scale_y
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_y
            skeletons.append(skeleton)
        return skeletons, scale_y

# ==============================================================

class MultiPersonClassifier(object):
    def __init__(self, LOAD_MODEL_PATH, action_labels):
        self.create_classifier = lambda human_id: ClassifierOnlineTest(
            LOAD_MODEL_PATH, action_types = action_labels, human_id=human_id)
        self.dict_id2clf = {} # human id -> classifier of this person
    '''
    def classify(self, dict_id2skeleton):

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():
            if id not in self.dict_id2clf: # add this new person
                self.dict_id2clf[id] = self.create_classifier(id)
            
            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton) # predict label

        return id2label
    '''
    def classify_ds(self, dict_id2skeleton, deepsort, number_of_person):

        # Clear people not in view

        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        numb = 0
        id2label = {}
        for ds_id, skeleton in dict_id2skeleton.items():

            if ds_id not in self.dict_id2clf:# and len(deepsort) == number_of_person: #and count < number_of_person: # add this new person
                self.dict_id2clf[ds_id] = self.create_classifier(deepsort[numb])
            
            #if len(deepsort) == len(dict_id2skeleton) and len(dict_id2skeleton) == len(self.dict_id2clf):#count < number_of_person: # add this new person TODO
            classifier = self.dict_id2clf[ds_id]
            id2label[ds_id] = classifier.predict(skeleton) # predict label
            numb += 1
            '''
            else:
                ds_id = previous_id + 1
                print("KeyError2", ds_id)
                classifier = self.create_classifier(ds_id)
                id2label[ds_id] = classifier.predict(skeleton) # predict label
                count += 1
            '''
        return id2label

    def get(self, id):
        # type: id: int or "min"
        if len(self.dict_id2clf) == 0:
            return None 
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def main():
############################################ DeepSort+YOLO    
   # Definition of the parameters
    max_cosine_distance = 0.1 # changed
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort 
    model_filename = 'D:/CTCI and UAV/OPENPOSE+DEEPSORT+YOLOv3/action_recognition_deepsort_id_switching/src/model_data/mars.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
############################################ 
    # -- Detect sekelton
    my_detector = SkeletonDetector(OpenPose_MODEL, image_size)
    yolo = YOLO()

    # -- Load images
    if FROM_FOLDER:
        images_loader = myio.DataLoader_folder(SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES)

    # -- Initialize human tracker and action classifier
    multipeople_classifier = MultiPersonClassifier(LOAD_MODEL_PATH, action_labels)
    multiperson_tracker = myfunc.Tracker()

    #video_capture = cv2.VideoCapture('D:/CTCI and UAV/OPENPOSE+DEEPSORT+YOLOv3/action_recognition_deepsort_yolo/src/example1.mp4')
    cap = choose_run_mode('D:/CTCI and UAV/OPENPOSE+DEEPSORT+YOLOv3/action_recognition_deepsort_id_switching/src/DJI_0140.mp4')
    video_writer = set_video_writer(cap, write_fps=int(2.0))
############################################# SmartWatch for two cases only
    sw_data1 = pd.read_csv("src/smartwatch/SmartWatch_1_person.csv") 
    sw_data2 = pd.read_csv("src/smartwatch/SmartWatch_2_person.csv")
    sw_i = 0
    action_smart_watch = {}

    # -- Loop through all images
    ith_img = 425#850#500#850#60#1#780#295#1
    while ith_img <= images_loader.num_images:
        img = images_loader.load_next_image()
        image_disp = img.copy()

        print("\n========================================")
        print("\nProcessing {}/{}th image\n".format(ith_img, images_loader.num_images))

        ################## DeepSort+YOLO
        image = Image.fromarray(image_disp[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)#image)
        features = encoder(image_disp, boxs)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
		# DeepSort
        tracker.predict()
        tracker.update(detections)

		# DeepSort white rectangle
        deepsort = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            deepsort.append(track.track_id) # append DEEP SORT ID
            bbox = track.to_tlbr()
            cv2.rectangle(image_disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            cv2.putText(image_disp, str(track.track_id), (int(bbox[0]) + 10, int(bbox[1]) - 10), 0, 5e-3 * 400, (0, 0, 255), 2)
        print("deepsort", deepsort)

        number_of_person = 0
		# YOLO blue rectangle
        for det in detections:
            number_of_person += 1
            bbox = det.to_tlbr()
            cv2.rectangle(image_disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        print("number_of_person", number_of_person)
        ##################

        # -- Detect all people's skeletons
        humans = my_detector.detect(img)
        skeletons, _ = my_detector.humans_to_skelsList(humans)

        # -- Track people
        dict_id2skeleton = multiperson_tracker.track(skeletons, deepsort, number_of_person)
        # -- Recognize action for each person
        
        if len(dict_id2skeleton) > 0 and len(deepsort) > 0: # there is at least one person
            if len(deepsort) == len(dict_id2skeleton): #number_of_person:
                dict_id2label = multipeople_classifier.classify_ds(dict_id2skeleton, deepsort, number_of_person)
                print("predicted label is :", dict_id2label)#, dict_id2label[min_id])
        
        # SmartWatch
        action_smart_watch[1] = sw_data1["STATUS"][sw_i]
        action_smart_watch[2] = sw_data2["STATUS"][sw_i]
        sw_i += 1
        print("predicted smart watch label is :", action_smart_watch)

        '''
        # -- Draw
        #image_disp = TfPoseEstimator.draw_humans(image_disp, humans, imgcopy=False) # Draw all skeletons
        #my_detector.draw(image_disp, humans) # Draw all skeletons
        if len(dict_id2skeleton) > 0 and len(deepsort) > 0: # there is a person
            # Draw outer box and label for each person 
            for ds_id, label in dict_id2label.items():
                if len(deepsort) == len(dict_id2skeleton): # and count < number_of_person:
                    try:
                        skeleton = dict_id2skeleton[ds_id]#deepsort[count]]
                        skeleton[1::2] = skeleton[1::2] / scale_y # scale the y data back to original
                        drawActionResult(image_disp, ds_id, skeleton, label, file1)
                    except KeyError:
                        print("KeyError")
        '''
        '''
        # -- Write skeleton.txt and image.png
        if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:
            cv2.imwrite(SAVE_DETECTED_SKELETON_IMAGES_TO 
                + myfunc.int2str(ith_img, 5) + ".png", image_disp)
        '''
        # -- Display
        if 1:
            if ith_img == 425:#1:#850:#500:#850:#60:#1:#780:#295:#1:
                window_name = "action_recognition"
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow(window_name, image_disp)
            video_writer.write(image_disp)
            q = cv2.waitKey(1)
            if q != -1 and chr(q) == 'q':
                break

        # -- Loop
        print("\n")
        ith_img += 1

    video_writer.release()
    cap.release()

if __name__ == '__main__':
    main()

