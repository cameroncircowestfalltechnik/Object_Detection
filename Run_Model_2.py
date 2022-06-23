#import libraries to load model
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppress TensorFlow logging
#import pathlib
import tensorflow as tf
import matplotlib.pyplot
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
matplotlib.use('TkAgg') #specify matplotlib backend becasue visualization_utils likes to break it 
tf.get_logger().setLevel('ERROR') #Suppress TensorFlow errors
import random


#-----Specify locations of important files-----
IMAGE_PATHS = "C:/Users/cameron.circo/Pictures/Beans/validation/angular_leaf_spot" #specify path for images to input (currently unused)
#specify some info about the model (unused)
MODEL_DATE = '20220621'
MODEL_NAME = 'my_ssd_resnet50_v1_fpn'

#specify where the model is saved (uses the most recent saved version of the model)
PATH_TO_MODEL_DIR = "C:/Users/cameron.circo/Documents/TensorFlow/models-master/workspace/training_demo2/exported-models/my_model"

#specify label filename and location
LABEL_FILENAME = 'label_map.pbtxt'
PATH_TO_LABELS = "C:/Users/cameron.circo/Documents/TensorFlow/models-master/workspace/training_demo2/annotations"

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model" #specify model folder

#-----Load the model and label map-----

print('Loading model...', end='') #report loading model
start_time = time.time() #reset timer to time model load

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL) #load the model

end_time = time.time() #stop the timer
elapsed_time = end_time - start_time #calculate elapsed time
print('Done! Took {} seconds'.format(elapsed_time)) #display it

#locate and extract labelmap
category_index = label_map_util.create_category_index_from_labelmap("C:/Users/cameron.circo/Documents/TensorFlow/models-master/workspace/training_demo2/annotations/label_map.pbtxt",
                                                                    use_display_name=True)

#-----Image Processing-----

#import libraries for holding and displaying images
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore') #Suppress Matplotlib warnings

image_path = 'C:/Users/cameron.circo/Pictures/test'
files = os.listdir(image_path)
qty = len(files)


for n in range(0,qty): #sweep through entry 0 to 42 doing the following:
    start_time = time.time()
    index = random.randint(0,qty-n)

    print("loading image") #display that image is being loaded


    #specify location of verification bad leaf "n"
    #image_path = "C:/Users/cameron.circo/Pictures/Beans/validation/angular_leaf_spot/angular_leaf_spot_val."+str(n)+".jpg"
    #specify location of verification healthy leaf "n"
    #image_path = "C:/Users/cameron.circo/Pictures/Beans/validation/healthy/healthy_val."+str(n)+".jpg"

    image = Image.open(image_path+'/'+files[index]) #open the image

    files.pop(index)
    image_np = np.array(image) #turn into image to array

    #--Sample Transformations--
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(image_np) #convert the image array to a tf tensor
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...] #add an axis (see line above)

    detections = detect_fn(input_tensor) #run the tensor through the model (this is where the magic happens)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections')) #removes "first" num_detections from detections and converts to interger
    detections = {key: value[0, :num_detections].numpy() #delete all detections
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections #reintroduce "first" num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64) #convert detection classes to intergers to be compatible with label map

    image_np_with_detections = image_np.copy() #copy the input image to a new array

    viz_utils.visualize_boxes_and_labels_on_image_array( #draw labels/boxes on top of copied image
            image_np_with_detections, #input image
            detections['detection_boxes'], #input box coords
            detections['detection_classes'], #input detection classes
            detections['detection_scores'], #input scores
            category_index, #input label map
            use_normalized_coordinates=True, #normalize coords
            max_boxes_to_draw=200, #specify max boxes
            min_score_thresh=.30, #specify score threshold
            agnostic_mode=False) #tell it not to ignore classes

    end_time = time.time() #stop the timer
    elapsed_time = end_time - start_time #calculate elapsed time
    print('Done! Took {} seconds'.format(elapsed_time)) #display it

    #print(image_np_with_detections) #debug print the the results

    plt.figure() #generate a figure
    plt.imshow(image_np_with_detections) #plot results

    data = Image.fromarray(image_np_with_detections) #generate results as image
    data.save('pic.png') #save image
    #plt.savefig('figure.png') #debug save figure as png
    plt.show() #push results to a window


