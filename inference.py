# USAGE
# python ncs_realtime_objectdetection.py --graph [path to graph] --display 1
# python ncs_realtime_objectdetection.py --graph [path to graph] --confidence 0.5 --display 1

# import the necessary packages
from firebase import firebase
import cv2, time
import numpy as np
import tensorflow as tf
import sys, os
import six.moves.urllib as urllib
import _thread
from distutils.version import StrictVersion
from collections import defaultdict
import tarfile, zipfile
import json
import argparse

"""
    REQUIRED:
        human_region.json
        frozen_model.pb
"""

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--verbose", default=1,help="show verbose")
ap.add_argument("-d", "--display", type=int, default=0, help="switch to display image on screen")
args = vars(ap.parse_args())

print("[INFO] initializing firebase")
firebase = firebase.FirebaseApplication('https://pi-movidius.firebaseio.com/', None)

def point2ratio(region):
    global w, h
    return region[0]/w, region[1]/h, region[2]/w, region[3]/h

def ratio2point(region):
    global w, h
    return int(region[0]*w), int(region[1]*h), int(region[2]*w), int(region[3]*h)

class VideoConnection(object):
    def __init__(self, rtsp_path, scale=1.0):
        self._rtsp_path = rtsp_path
        self._cap = cv2.VideoCapture(rtsp_path)
        self._cap.set(3, 1)
        self._scale = scale

    def get_lastest(self):
        self._cap.set(1,-1)
        ret, frame = self._cap.read()
        if self._scale != 1.0:
            cv2.resize(frame, (0,0), fx=self._scale, fy=self._scale)
        return ret, frame

    def get(self, skip=0, wait_for_skip=0):
        while True:
            try:
                ts = time.time()
                img = self._cap.grab()
                for i in range(skip):
                    img = self._cap.grab()
                    if time.time() - ts > wait_for_skip:
                        break
                ret, frame = self._cap.retrieve()
                h, w, _ = frame.shape
                if self._scale != 1.0:
                    frame = cv2.resize(frame, (0,0), fx=self._scale, fy=self._scale)
                break
            except:
                print("Video Connection lost. Trying to reconnect the camera...")
                self._cap = cv2.VideoCapture(self._rtsp_path)
        return ret, frame

    def match_region(self, centers, regions):
        position_sets = []
        for c in centers:
            position_sets.append([])
            for r in range(len(regions)):
                rg = regions[r]
                if rg[0] < c[0] and c[0] < rg[2] and rg[1] < c[1] and c[1] < rg[3]:
                    position_sets[-1].append(r)
        return position_sets

class TFAPI(object):
    def __init__(self):
        self._load_model()

    def _load_model(self):
        with tf.device("/device:GPU:0"):
            GRAPH_PATH = './frozen_model.pb'
            self._net = tf.Graph()
            with self._net.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

    def predict(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        with self._net.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'out/Sigmoid'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                image_tensor = tf.get_default_graph().get_tensor_by_name('input:0')
                # training = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')
#                     print(sess.run(tensor_dict['num_detections'], feed_dict={image_tensor: image}))
#                     # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                out = output_dict['out/Sigmoid']
            return out

class Cropper(object):
    def __init__(self, boxes, patch_size=None): # [[x1_1, y1_1, x1_2, y1_2], [x2_1, y2_1, x2_2, y2_2], ...]
        self._boxes = boxes
        self._patch_size = patch_size

    def crop(self, images):
        self._size = (images.shape[1], images.shape[2])
        patches = []
        for img in images:
            for b in self._boxes:
                x1, y1, x2, y2 = ratio2point(b)
                cropped = img[int(y1):int(y2), int(x1):int(x2), :]
                if self._patch_size is not None:
                    cropped = cv2.resize(cropped, self._patch_size)
                patches.append(cropped)
        return np.concatenate([patches], axis=0)

print("[INFO] Loading human region")
if not os.path.exists("./human_region.json"):
    f = open("human_region.json", "w+")
    json.dump({"bound": [], "name": []}, f)
    f.close()
    region = []
    annotation = []
    name = []
else:
    # load data when the file exists
    f = open("human_region.json")
    data = json.load(f)
    region = data["bound"]
    annotation = data["name"]
    f.close()

batch_size = 10
cache_imgs = []
cache_time = []
i = 0
h = 0
print("[INFO] Loading graph")
net = TFAPI()
print("[INFO] Connect to RTSP or VIDEO")
cam = VideoConnection("../NVR_ch3_main_20180826150000_20180826160000.mp4", scale=1)

def post_data(object):
	result = firebase.post('detection', object)
	return result 


print("[INFO] Start Prediction")
while(True):
    time_start = time.time() # start time
    ret, img = cam.get(skip=50, wait_for_skip=0.1)
    imgs = np.expand_dims(img, axis=0)
    _, h, w, _ = imgs.shape
    crp = Cropper(region, patch_size=(50, 50))
    cropped = crp.crop(imgs)
    a = time.time()
    out = net.predict(cropped)
    ptime = time.time() - a
    detect_centers = []
    detect_results = [[]]
    for k in range(len(region)):
        if (out[k])[0] > threshold:
            detect_results[0].append(k)
        confidence[0][k] = inc_conf(confidence[0][k], int((out[k])[0] > threshold))
    result = np.expand_dims(np.where(np.array(confidence) > threshold)[1], -1).tolist()
    print(result)

    time_end = time.time()
    infer_time = time_end - time_start

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    json_data = {
        "count": len(result),
        "position": result,
        "catagories": result,
        "catagory_names": annotation,
        "timestamp": st,
        "having_human": confidence[0],
        "infer_time": ptime,
        "overall_time": infer_time,
    }

    if args["verbose"] == 1:
        print("[DATA]",json_data)
    if push_firebase == 1 and ref_time - last_push_fb > FLAGS.send_fb_time:
            last_push_fb = ref_time
            firebase.push_data(json_data)
    post_data(json_data)
    

    if (True):
        _p = 0
        for k in range(len(region)):
            if (out[_p])[0] > threshold:
                xmin, ymin, xmax, ymax = ratio2point(region[k])
                cv2.rectangle(imgs[0], (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            _p += 1

    if args["display"] == 1:
        cv2.imshow('frame', imgs[0])
        K = cv2.waitKey(1)
        if K & 0xFF == ord('q'):
            break

