import os

from detectionUtilities.preprocessing import frame_segregation

if __name__ == '__main__':
    src_path = "/home/rv/Documents/datasets/traffic/custom/hiv00031-frames"
    frame_segregation(src_path)
