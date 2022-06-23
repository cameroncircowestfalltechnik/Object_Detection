# Object_Detection
# Introduction
This page is kind of loose documentation for my object detection project. Please at least read the whole paragraph attached to each entry to ensure it runs right.  
Primary sources should be [The official Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records) and [Gilbert Tanner's object Arduino Detector](https://gilberttanner.com/blog/creating-your-own-objectdetector/). These sources are probably a lot better than this page.  
The official documentation is mostly what I used, however Gilbert Tanner's documentation explains some of the steps better.  

## Tools:  
[beans data set (I removed the rusted ones)](https://www.tensorflow.org/datasets/catalog/beans)  
[Labelimg (follow official documentation here to install)](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#use-precompiled-binaries-easy)  
Windows visual studio 2022  
  
## Setup
Start by generating a labelmap file  
  -I added mine in the annotations folder as a sample  
  -This file type is to be modified in visual studio  

After setting everything up and using labelimg on your data I use **generate_tfrecord.py**  
  -This program takes the the label map you made and the xml files and combines them into a .record file  
  -Important! (make sure “class_text_to_int” is up to date with current label names (~line 104)  
  -It is executed as so:  
```
cd [path to generate_tfrecord.py]
py generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record  
py generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record  
```
Next you can run **Read_TFRecord** to make sure that it wrote correctly, I executed this in visual studio and obviously you will have to route it to your .record file  
  -feel free to use the test.record under annotations to check it is populations correctly  

## Training
Next run **model_main_tf2.py** to train the model in my case I saved it in the "root folder" as specified by the official documentation  
  -Be sure that pipeline.config is up to date with the correct label qty and directories, again I uploaded mine as an example  
  -model_main_tf2.py is run by doing the following  
```
cd [path to "root folder"]  
py model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config  
```
-Running this program should be extremely intensive (it ran my cpu to about 90 percent usage) on my work laptop the first batch of 100 operations took ~45 seconds each meaning it took about 68 minutes before posting anything to the console. In my case I remoted into my gaming setup at home with a gtx 1070 and ran the training. This lowered my operation time to around 1.6 seconds but is still a serious load on the pc, it immediately used all the vram  

Finally you can export the model using **exporter_main_v2.py**, this step is also pretty computationally taxing and should be run with the training closed, ideally soon after it posts a checkpoint. In my case I forgot this and ended up running my ram, ssd, cpu, and vram all at 100 percent usage. 
 -This should export your model into the exported-models directory  
 -use like so (while still cd'd into the "root folder"):  
 ```
    py ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir ./models/my_ssd_resnet50_v1_fpn/ --output_directory ./exported-models/my_model  
```
## Running the model    
Now you can use your model, in my case this is done in **Run_Model_2.py**, a modified version of the sample program "Object Detection From TF2 Checkpoint" from the official documentation
-This is run within visual studio  
-The comments inside the program are the best source of documentation so read them for more info and step by step insight  
-I mainly modified it to pull the images, model, label map, etc. from the disk as well as modified it to load random images from the verification folder  
-This program should load everything in (this takes 60-100 seconds) and then load up an image with the box and label attached (see sample_output.png), if it is not adding a label, it is not certain enough to add a prediction and your model will need more training  
