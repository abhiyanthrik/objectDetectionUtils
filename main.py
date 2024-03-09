import os
from detectionUtilities import create_empty_annotations


if __name__ == '__main__':
    images_path = '/home/s2p/Documents/COCO/folders/dataset/val2017/images'
    labels_path = '/home/s2p/Documents/COCO/folders/dataset/val2017/labels'
    # dst_dir = '/home/s2p/Documents/COCO/folders/dataset/train2017-filtered'
    create_empty_annotations(images_path, labels_path)
