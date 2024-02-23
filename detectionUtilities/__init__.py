import os
import shutil
import xml.etree.ElementTree as ETree
from typing import List
import random
from xml.etree.ElementTree import Element
classes = ['pistol', 'knife']


def create_dataset(root_dir: str, data_format: str = 'coco', splits: (float, float) = (0.2, 0.05)):
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    label_ext = '.xml' if data_format == 'coco' else '.txt' if data_format == 'yolo' else None
    if label_ext is None:
        print('Unknown Dataset to split...\nTerminating the operation!!!')
        return
    dataset_dir = f"{root_dir}-dataset"
    print(f"Creating {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)
    files = os.listdir(os.path.join(root_dir, "images"))
    files = [os.path.splitext(file)[0] for file in files]
    for _ in range(25):
        random.shuffle(files)

    data_size = len(files)
    train_till = int(data_size * (1 - sum(splits)))
    valid_till = int(train_till + (data_size * splits[0]))
    dataset_splits = {
        "train": files[:train_till],
        "valid": files[train_till:valid_till],
        "test": files[valid_till:]
    }
    for split in dataset_splits:
        split_dir = os.path.join(dataset_dir, split)
        print(f"\tCreating {split_dir}")
        for file in dataset_splits[split]:
            image_path = ''
            for image_ext in image_extensions:
                image_name = f"{file}{image_ext}"
                image_path = os.path.join(root_dir, "images", image_name)
                if os.path.exists(image_path):
                    break
            if not os.path.exists(image_path):
                print(f"\t\tSkipping {image_path}...\n\t\tFile not exists!!!")
                continue
            label_name = f"{file}{label_ext}"
            label_path = os.path.join(root_dir, "labels", label_name)
            image_dest, label_dest = [os.path.join(split_dir, dest) for dest in ["images", "labels"]]
            for dest in [image_dest, label_dest]:
                if not os.path.exists(dest):
                    os.makedirs(dest, exist_ok=True)
            print(f"\t\tCopying: {image_path}")
            shutil.copy2(image_path, image_dest)
            print(f"\t\tCopying: {label_path}")
            shutil.copy2(label_path, label_dest)


def merge_data(src_paths: List[str], dst_path: str):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    if os.listdir(dst_path):
        print(f'{dst_path} Not empty!!!\nChoose a different location')
        return
    for src_path in src_paths:
        print(f"Copying {src_path}")
        for root, dirs, files in os.walk(src_path):
            for file in files:
                file_abs_path = os.path.join(root, file)
                file_rel_path = file_abs_path.replace(src_path+'/', '')
                dest_path = "/".join(os.path.join(dst_path, file_rel_path).split('/')[:-1])
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                print(f"\tCopying {file_abs_path}")
                shutil.copy2(file_abs_path, dest_path)


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
        print(f"Working on: {subdirectory}")
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


def overlay_boxes(image, bounding_boxes):
    pass


def get_box_from_yolo(image_src, label_src):
    pass


def draw_bounding_boxes(image_path, label_path):
    pass
