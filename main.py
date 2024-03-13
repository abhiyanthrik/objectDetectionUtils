import os
from detectionUtilities import filter_classes, validate_annotations


if __name__ == '__main__':
    src_path = '/home/s2p/Documents/COCO/folders/dataset-copy'
    dst_path = '/home/s2p/Documents/COCO/folders/person-filtered'
    classes = ['person']
    filter_classes(src_path, dst_path, classes)
    validate_annotations("/home/s2p/Documents/COCO/folders/dataset-filtered/labels")
