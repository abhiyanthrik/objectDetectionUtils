import os
from detectionUtilities import filter_classes, create_dataset, voc2yolo_single, delete_classes, validate_annotations


if __name__ == '__main__':
    # src_dir = '/home/s2p/Documents/datasets/weapons-merged-yolo2702'
    # dst_dir = src_dir + '-yolo2702'
    # voc2yolo_single(src_dir, dst_dir)
    # create_dataset(src_dir, 'yolo')
    # sources = [os.path.join(src_dir, src) for src in os.listdir(src_dir)]
    # merge_data(sources, dst_dir)
    # voc2yolo(src_dir, dst_dir)
    images_path = '/home/s2p/Documents/datasets/fire-data-roboflow-2802/fire-data/images'
    labels_path = '/home/s2p/Documents/datasets/fire-data-roboflow-2802/fire-data/labels'
    cls_ids = [1]  # [24, 26]
    dst_dir = '/home/s2p/Documents/datasets/fire-data-roboflow-2802/fire-data-filtered'
    # delete_classes(images_path, labels_path, dst_dir, cls_ids)
    validate_annotations('/home/s2p/Documents/datasets/fire-data-roboflow-2802/fire-data-filtered/labels')
