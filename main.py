import os

from detectionUtilities import voc2yolo, merge_data, create_dataset


if __name__ == '__main__':
    src_dir = '/home/s2p/PycharmProjects/dataPreprocess/data/weapons-merged-dataset'
    dst_dir = src_dir + '-yolov1'
    # sources = [os.path.join(src_dir, src) for src in os.listdir(src_dir)]
    # merge_data(sources, dst_dir)
    voc2yolo(src_dir, dst_dir)
