import os
from detectionUtilities import voc2yolo


if __name__ == '__main__':
    src_path = '/home/s2p/PycharmProjects/dataPreprocess/data/archive'
    dst_path = src_path + '-yolo'
    voc2yolo(src_path, dst_path)
