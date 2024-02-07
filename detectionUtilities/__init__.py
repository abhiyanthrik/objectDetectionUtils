import os
import shutil
import xml.etree.ElementTree as ETree
from xml.etree.ElementTree import Element

classes = ['person']


def merge_data(src_path, dst_path):
    pass


def copy_files(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file in os.listdir(source_dir):
        if os.path.exists(os.path.join(dest_dir, file)):
            print(f"Skipping {file}...\talready exists in {dest_dir}!")
            continue
        shutil.copy2(source_dir + '/' + file, dest_dir)


def normalize_bounding_boxes(size, bounding_boxes):
    dw, dh = 1. / (size[0]), 1. / (size[1])
    x = ((bounding_boxes[0] + bounding_boxes[1]) / 2.0 - 1) * dw
    y = ((bounding_boxes[2] + bounding_boxes[3]) / 2.0 - 1) * dh
    w = (bounding_boxes[1] - bounding_boxes[0]) * dw
    h = (bounding_boxes[3] - bounding_boxes[2]) * dh
    return x, y, w, h


def get_bounding_boxes(xml_root: Element):
    w, h = [int(xml_root.find('size').find(tag).text) for tag in ('width', 'height')]
    boxes = {}
    for obj in xml_root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        bounding_box = [float(xml_box.find(tag).text) for tag in ('xmin', 'xmax', 'ymin', 'ymax')]
        normalized_bounding_box = normalize_bounding_boxes((w, h), bounding_box)
        boxes[cls_id] = normalized_bounding_box
    return boxes


def move_content(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file in os.listdir(src_dir):
        in_file = os.path.join(src_dir, file)
        out_file = os.path.join(dest_dir, f"{os.path.splitext(file)[0]}.txt")
        if os.path.exists(out_file):
            print(f"Skipping {in_file}...\tAlready exists!")
            continue
        with open(in_file, 'r') as in_file:
            tree = ETree.parse(in_file)
            root = tree.getroot()
            bounding_boxes = get_bounding_boxes(root)
        if os.path.exists(out_file):
            continue
        with open(out_file, 'w') as out_file:
            for box in bounding_boxes:
                out_file.write(f"{box} {' '.join([str(a) for a in bounding_boxes[box]])}\n")


def voc2yolo(src_dir, dest_dir):
    for subdirectory in os.listdir(src_dir):
        src_split = os.path.join(src_dir, subdirectory)
        split_name = src_split.replace(src_dir, '')
        dst_split = dest_dir + split_name
        src_split_content = os.listdir(src_split)
        for content in src_split_content:
            src_content_path = os.path.join(src_split, content)
            dst_content_path = os.path.join(dst_split, content)
            if 'image' in content.lower():
                copy_files(src_content_path, dst_content_path)
            elif 'label' in content.lower():
                move_content(src_content_path, dst_content_path)