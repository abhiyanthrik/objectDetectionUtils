import os
from detectionUtilities import draw_bounding_boxes

if __name__ == '__main__':
    src_dir = '/home/s2p/PycharmProjects/dataPreprocess/data/weapons-merged-dataset'
    dst_dir = src_dir + '-yolov1'
    # sources = [os.path.join(src_dir, src) for src in os.listdir(src_dir)]
    # merge_data(sources, dst_dir)
    # voc2yolo(src_dir, dst_dir)
    images_path = '/home/rv/Documents/COCO/dirs/val2017'
    labels_path = '/home/rv/Documents/datasets/new_dir/labels/val2017'
    for filename in os.listdir(images_path)[:10]:
        img_path = os.path.join(images_path, filename)
        filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(labels_path, filename)
        draw_bounding_boxes(img_path, label_path)
