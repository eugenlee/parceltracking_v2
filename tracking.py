from detection_helpers import *
from tracking_helpers import *
from bridge_wrapper import *
from PIL import Image

detector = Detector(iou_thresh=0.6,conf_thres=0.3) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./weights/best2.pt') # pass the path to the trained weight file

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
tracker.track_video(0, output="./IO_data/output/yeah.avi", show_live = True, skip_frames = 0, count_objects = True, verbose=1)


# self.conf_thres = conf_thres
# self.iou_thres = iou_thresh
# self.classes = classes
# self.agnostic_nms = agnostic_nms
# self.save_conf = save_conf

# conf_thres: Thresholf for Classification
# iou_thres: Thresholf for IOU box to consider
# agnostic_nms: whether to use Class-Agnostic NMS
# save_conf: whether to save confidences in 'save_txt' labels afters inference
# classes: Filter by class from COCO. can be in the format [0] or [0,1,2] etc