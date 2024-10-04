import cv2

from detectionUtilities import combine_data_v2, draw_bounding_boxes

if __name__ == '__main__':
    src_path = "/home/rohit/security/weapon"
    # combine_data_v2(src_path)
    image_path = "/home/rohit/security/weapon-combined/person/images/-8_jpg.rf.8387ae521dc1600b8995f06fff65b9ec.jpg"
    label_path = "/home/rohit/security/weapon-combined/person/labels/-8_jpg.rf.8387ae521dc1600b8995f06fff65b9ec.txt"
    drawn = draw_bounding_boxes(image_path, label_path)
    cv2.imshow("drawn", drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
