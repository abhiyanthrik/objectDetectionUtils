import os
import shutil

data_path = '/home/rv/Documents/datasets/new-number-dataset-26-03-copy'
annotation_path = os.path.join(data_path, 'train/annotations.txt')

with open(annotation_path, 'r') as annotation_file:
    annotations = annotation_file.readlines()

for annotation in annotations:
    annotation = annotation.strip().split()
    print(annotation)
