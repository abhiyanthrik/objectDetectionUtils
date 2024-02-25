import os
from detectionUtilities import filter_classes, draw_bounding_boxes


if __name__ == '__main__':
    # src_dir = '/home/s2p/PycharmProjects/dataPreprocess/data/weapons-merged-dataset'
    # dst_dir = src_dir + '-yolov1'
    # sources = [os.path.join(src_dir, src) for src in os.listdir(src_dir)]
    # merge_data(sources, dst_dir)
    # voc2yolo(src_dir, dst_dir)
    images_path = '/home/rv/Documents/COCO/dirs/val2017'
    labels_path = '/home/rv/Documents/datasets/new_dir/labels/val2017'
    dst_dir = '/home/rv/Documents/datasets/human-dataset-from-coco'
    for filename in os.listdir(images_path):
        img_path = os.path.join(images_path, filename)
        filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(labels_path, filename)
        if not os.path.exists(label_path):
            with open(label_path, 'w') as _:
                print(label_path)

        # draw_bounding_boxes(img_path, label_path)
    print(len(os.listdir(labels_path)), len(os.listdir(images_path)))
    # images_path = '/home/rv/Documents/COCO/dirs/train2017'
    # labels_path = '/home/rv/Documents/datasets/new_dir/labels/val2017'
    cls_ids = [0]  # [24, 26]
    filter_classes(images_path, labels_path, dst_dir, cls_ids)
