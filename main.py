import os
from datetime import datetime
from threading import*
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import PIL
import cv2 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import pyrebase
from pyrebase.pyrebase import Storage

class Window():
    def __init__(self):
        self.text=""
        self.label= tk.Label(self.root, text = self.text, width=200, height=300, fg="blue")
        self.label.pack() 

def web():
    cap = cv2.VideoCapture(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.4,
                    agnostic_mode=False)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (500, 500)))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

def showimage():    
    fln =filedialog.askopenfilename(initialdir=os.getcwd(), title= "Select Image File", filetypes=(("PNG file", "*.png"),("JPEG file","*.jpeg" ), ("All files", "*.")))
    imgM = fln
    analizeimage(imgM)

         
def analizeimage(imgM):
    matplotlib.use('TkAgg')
    # %%matplotlib inline
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    #print(imgM)
    img = cv2.imread(imgM)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    aa=viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=2,
                min_score_thresh=.90,
                agnostic_mode=False
                )
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    boxes = detections['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = detections['detection_scores']
    min_score_thresh=.90
    class_name=""
    ns=""
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            class_name = category_index[detections['detection_classes'][i]+1]['name']
            ns=scores[i]        
            break
    strtime=datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    path_on_cloud="DataSet/"+ strtime + "_"+ class_name +".png"
    
    import io
    plt.axis('off')
    buf =io.BytesIO()
    plt.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img=PIL.Image.open(buf)
    img.thumbnail((450,450))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image =img

    txt=""

    text=txt=class_name +": "+  str(ns)
    print(txt)
    labelT.config(text=txt)
    
    uploadimage(imgM, path_on_cloud)
   
 
#@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def uploadimage(fln, path_on_cloud):

    config = {
        "apiKey": "AIzaSyBbGSXMqY1W9_eS9jWc7WrUM7KWlDHSeM8",

        "authDomain": "saved-img.firebaseapp.com",

        "projectId": "saved-img",

        "storageBucket": "saved-img.appspot.com",

        "messagingSenderId": "41701132620",

        "appId": "1:41701132620:web:f1de2e8717e39a2d7527e9",

        "measurementId": "G-FQK562DE6T",
        
        "serviceAccount": "service-Account.json",
        
        "databaseURL": "https://saved-img-default-rtdb.firebaseio.com/"
    }
    
    firebase = pyrebase.initialize_app(config)
    Storage = firebase.storage()
    Imagen = fln
    Storage.child(path_on_cloud).put(Imagen)

if __name__ == "__main__":
   
    CUSTOM_MODEL_NAME = 'melanoma_ssd_mobnet_57' 
    LABEL_MAP_NAME = 'label_map.pbtxt'
    paths = {
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    }

    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    # Load pipeline config and bouild a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model =  detection_model = model_builder.build(model_config=configs['model'], is_training=False)
   
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-8')).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    
    root = Tk()
    frm = Frame(root)
    frm.pack(side=BOTTOM,padx=10, pady=10)
    
    lbl = Label(root)
    lbl.pack()

    btn = Button(frm, text ="Analize Image", command = showimage)
    btn.pack(side=tk.LEFT)
    
    btn = Button(frm, text ="Live Feed", command = web)
    btn.pack(side=tk.RIGHT)

    labelT= tk.Label(root, text = "", width=100, height=100, fg="blue")
    labelT.pack()
    
    root.title("Lunar Detector")
    root.geometry("600x500")
    
    root.mainloop()