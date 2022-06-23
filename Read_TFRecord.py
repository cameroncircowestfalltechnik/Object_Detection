import tensorflow as tf
import numpy
import os

path = "C:/Users/cameron.circo/Documents/TensorFlow/models-master/workspace/training_demo2/annotations/train.record"

print("Debug")

raw_dataset = tf.data.TFRecordDataset(path)

print(raw_dataset)
print("start2")
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

    