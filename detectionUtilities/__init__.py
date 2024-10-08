import yaml
import json
import os
import random
import shutil
import xml.etree.ElementTree as ETree
from typing import List, Dict
from xml.etree.ElementTree import Element

import cv2

classes = ['autorickshaw', 'knife']


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
                file_rel_path = file_abs_path.replace(src_path + '/', '')
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
    classes = {
        'autorickshaw': 0,
        'tractor': 1,
        'truck': 2,
        'Rikshaw': 3,
        'Rickshaw': 3,
        'rikshaw': 3
    }
    w, h = [int(xml_root.find('size').find(tag).text) for tag in ('width', 'height')]
    boxes = {}
    for obj in xml_root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        print('\t', cls)
        if cls not in classes.keys() or int(difficult) == 1:
            continue

        cls_id = classes[cls]
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


def voc2yolo_single(src_dir, dest_dir):
    for content in os.listdir(src_dir):
        src_content_path = os.path.join(src_dir, content)
        dst_content_path = os.path.join(dest_dir, content)
        if 'image' in content.lower():
            copy_files(src_content_path, dst_content_path)
        elif 'label' in content.lower():
            move_content(src_content_path, dst_content_path)


def seg_to_bbox(seg_info):
    class_id = seg_info[0]
    points = seg_info[1:]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    bbox_info = class_id, x_center, y_center, width, height
    return bbox_info


def get_box_from_yolo(image_shape, yolo_box):
    height, width = image_shape
    x, y, w, h = yolo_box
    box_width = width * w
    box_height = height * h
    x *= width
    y *= height
    x0 = x - box_width / 2
    x1 = x0 + box_width
    y0 = y - box_height / 2
    y1 = y0 + box_height
    return x0, y0, x1, y1


def overlay_boxes(image, bounding_boxes: List[Dict]):
    height, width = image.shape[0], image.shape[1]
    for box in bounding_boxes:
        label = str(box['label'])
        bbox = box['bbox']
        bbox = get_box_from_yolo((height, width), bbox)
        start = int(bbox[0]), int(bbox[1])
        end = int(bbox[2]), int(bbox[3])
        origin = (start[0] - 15, start[1] - 15)
        image = cv2.rectangle(image, start, end, (0, 0, 255), 2)
        image = cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
    return image


def draw_bounding_boxes(image_path, label_path, segmented=False):
    image = cv2.imread(image_path)
    box_info = []
    if not os.path.exists(label_path):
        with open(label_path, 'w') as _:
            ...
    with open(label_path, 'r') as label_file:
        lines = label_file.read().splitlines()
        for line in lines:
            box = {}
            annotation = line.split()
            annotation[0] = int(annotation[0])
            annotation[1:] = [float(a) for a in annotation[1:]]
            if segmented:
                annotation = seg_to_bbox(annotation)
            # print(line)
            # print(annotation, '\n')
            box['label'] = annotation[0]
            box['bbox'] = annotation[1:]
            box_info.append(box)
    drawn = overlay_boxes(image, box_info)
    return drawn


def create_data(image_path: str, label_path: str, dst_dir: str, class_ids) -> None:
    file_name = os.path.basename(label_path)
    dst_img_path = os.path.join(dst_dir, 'images')
    if not os.path.exists(dst_img_path):
        os.makedirs(dst_img_path)
    dst_label_path = os.path.join(dst_dir, 'labels')
    if not os.path.exists(dst_label_path):
        os.makedirs(dst_label_path)

    shutil.copy2(image_path, dst_img_path)
    src_ann = open(label_path, 'r')
    dst_ann_file = open(os.path.join(dst_label_path, file_name), 'w')
    lines_of_interest = []
    for line in src_ann.readlines():
        line_ = line.split()
        cls_id = int(line_[0])
        # bbox = list(seg_to_bbox(line_))
        if cls_id in class_ids:
            line_[0] = str(class_ids.index(cls_id))
            bb = ' '.join(line_)
            lines_of_interest.append(bb + '\n')
    dst_ann_file.writelines(lines_of_interest)


def filter_classes(src_path: str, dst_path: str, class_names: List[str]) -> None:
    if os.path.exists(dst_path):
        print("Destination Directory already exist...\nPlease chose a different name.")
        return
    os.makedirs(dst_path, exist_ok=True)
    annotation_path = os.path.join(src_path, 'labels')
    image_path = os.path.join(src_path, 'images')
    metadata_path = os.path.join(src_path, 'data.yaml')
    metadata_file = open(metadata_path, 'r')
    original_classes = json.load(metadata_file)['nc']
    metadata_file.close()
    class_ids = []
    for cls in class_names:
        if cls not in original_classes:
            print('Class {} not found in original classes. Please choose a valid class.'.format(cls))
            return
        class_ids.append(original_classes.index(cls))
    new_metadata = {
        'path': dst_path,
        'nc': [original_classes[i] for i in class_ids]
    }
    new_metadata_filename = os.path.join(dst_path, 'data.yaml')
    with open(new_metadata_filename, 'w') as f:
        json.dump(new_metadata, f)

    for annotation_file in os.listdir(annotation_path):
        ann_file_path = os.path.join(annotation_path, annotation_file)
        img_file_name = os.path.splitext(annotation_file)[0] + '.jpg'
        img_file_path = os.path.join(image_path, img_file_name)

        with open(ann_file_path, 'r') as ann:
            lines = ann.read().splitlines()
            for line in lines:
                line_ = line.split()
                cls = int(line_[0])
                if cls in class_ids:
                    print(f'filtering for {ann_file_path}')
                    create_data(img_file_path, ann_file_path, dst_path, class_ids)
                    break


def delete_data(img_path: str, ann_path: str, dst_path: str, cls_ids: list) -> None:
    file_name = os.path.basename(ann_path)
    dst_img_path = os.path.join(dst_path, 'images')
    if not os.path.exists(dst_img_path):
        os.makedirs(dst_img_path)
    dst_label_path = os.path.join(dst_path, 'labels')
    if not os.path.exists(dst_label_path):
        os.makedirs(dst_label_path)

    shutil.copy2(img_path, dst_img_path)

    src_ann = open(ann_path, 'r')
    dst_ann_file = open(os.path.join(dst_label_path, file_name), 'w')
    lines_of_interest = []
    for line in src_ann.readlines():
        line_ = line.split()
        cls_id = int(line_[0])
        if cls_id in cls_ids:
            continue
        if cls_id == 2:
            line_[0] = '1'
        bb = ' '.join(line_)
        lines_of_interest.append(bb + '\n')
    dst_ann_file.writelines(lines_of_interest)


def delete_classes(image_path: str, annotation_path: str, dst_path: str, class_ids: list) -> None:
    for annotation_file in os.listdir(annotation_path):
        ann_file_path = os.path.join(annotation_path, annotation_file)
        img_file_name = os.path.splitext(annotation_file)[0] + '.jpg'
        img_file_path = os.path.join(image_path, img_file_name)
        delete_data(img_file_path, ann_file_path, dst_path, class_ids)


def filter_unannotated(image_path: str, annotation_path: str, dst_path: str):
    images = os.listdir(image_path)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    for img in images:
        img_path = os.path.join(image_path, img)
        label_path = os.path.join(annotation_path, f"{os.path.splitext(img)[0]}.txt")
        if not os.path.exists(label_path):
            print(f"Copying file {img_path} to {dst_path}")
            shutil.copy2(img_path, dst_path)
            # print(img_path)


def create_empty_annotations(image_path: str, annotation_path: str):
    images = os.listdir(image_path)
    for img in images:
        img_path = os.path.join(image_path, img)
        label_path = os.path.join(annotation_path, f"{os.path.splitext(img)[0]}.txt")
        if not os.path.exists(label_path):
            print(f"creating label file {label_path}")
            with open(label_path, 'w') as _:
                pass


def validate_annotations(annotation_path: str):
    cls_of_interest = [0]
    for ann_file in os.listdir(annotation_path):
        ann_file = os.path.join(annotation_path, ann_file)
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("Empty file")
            for line in lines:
                line = line.split()
                if int(line[0]) not in cls_of_interest:
                    print(f"Invalid annotation file: {ann_file}")


def segregate_xml_from_images(dataset_path: str) -> None:
    dataset = {i: os.path.join(f'{dataset_path}-sagregated', i) for i in ("images", "annotations")}
    for d in dataset.values():
        print(os.path.join(dataset_path, d))
        os.makedirs(os.path.join(dataset_path, d), exist_ok=True)

    for file in os.listdir(dataset_path):
        src_file_path = os.path.join(dataset_path, file)
        if file.endswith(".xml"):
            dst_path = os.path.join(dataset['annotations'], file)
            print(f"Moving: {file} to {dataset['annotations']}")
            shutil.move(src_file_path, dst_path)
            # print(f"src: {src_file_path}")
            # print(f"dst: {dst_path}\n")
        else:
            dst_path = os.path.join(dataset['images'], file)
            print(f"Moving: {file} to {dataset['images']}")
            shutil.move(src_file_path, dst_path)
            # print(f"src: {src_file_path}")
            # print(f"dst: {dst_path}\n")


def classes_from_voc(dataset_path: str):
    classes = []
    data_dirs = os.listdir(dataset_path)
    for data_dir in data_dirs:
        label_path = os.path.join(dataset_path, data_dir, 'labels')
        if not os.path.exists(label_path):
            print(f"The Path: {label_path} not found")
            continue
        label_files = os.listdir(label_path)
        for label_file in label_files:
            file_path = os.path.join(label_path, label_file)
            if os.path.isdir(file_path):
                print(f"Skipping {label_file} [It's a directory]")
                continue
            with open(file_path, 'r') as f:
                tree = ETree.parse(f)
                root = tree.getroot()
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        classes.append(cls)
    return classes


def combine_datasets(dataset_path: str) -> None:
    combined_data_path = f'{dataset_path}-combined'
    if not os.path.exists(combined_data_path):
        os.makedirs(combined_data_path, exist_ok=True)
    if os.listdir(combined_data_path):
        print(f"The Path: {combined_data_path} is not empty, please check content and remove them")
    combined_classes = dict()
    weapon_classes = ['weapon', 'dangerous weapon', 'Pistol', 'knife', 'Knife', 'pistol', 'rifle']
    human_classes = ['criminal', 'person', 'Person', 'victim']
    data_dirs = os.listdir(dataset_path)
    for data_dir in data_dirs:
        data_path = os.path.join(dataset_path, data_dir)
        meta_file = os.path.join(data_path, 'data.yaml')
        with open(meta_file, 'r') as f:
            classes = yaml.safe_load(f)['names']
        for cls in classes:
            if cls not in combined_classes:
                combined_classes[cls] = len(combined_classes)
        # images_path = os.path.join(data_path, 'images')
        # labels_path = os.path.join(data_path, 'labels')
        #
        # dest_images_path = os.path.join(combined_data_path, 'images')
        # dest_labels_path = os.path.join(combined_data_path, 'labels')
        #
        # os.makedirs(dest_images_path, exist_ok=True)
        # os.makedirs(dest_labels_path, exist_ok=True)
        #
        # image_list = os.listdir(images_path)
        # for image in image_list:
        #     image_name = os.path.splitext(image)[0]
        #     src_image_path = os.path.join(images_path, image)
        #     print(f"Copying image {image} to {dest_images_path}")
        #     shutil.copy2(src_image_path, dest_images_path)
        #
        #     label_name = f"{image_name}.txt"
        #     src_label_path = os.path.join(labels_path, label_name)
        #     dest_label_path = os.path.join(dest_labels_path, label_name)
        #     with open(src_label_path, 'r') as original_annotation:
        #         annotations = original_annotation.readlines()
        #     annotations = [annotation.split() for annotation in annotations]
        #     for annotation in annotations:
        #         annotation[0] = str(combined_classes[classes[int(annotation[0])]])
        #     annotations = [' '.join(annotation) for annotation in annotations]
        #     annotations = '\n'.join(annotations)
        #     with open(dest_label_path, 'w') as final_annotation:
        #         final_annotation.write(annotations)

    combined_cls = [None] * len(combined_classes)
    for cls in combined_classes:
        combined_cls[combined_classes[cls]] = cls


def combine_data_v2(dataset_path: str) -> None:
    weapon_classes = ['weapon', 'dangerous weapon', 'Pistol', 'knife', 'Knife', 'pistol', 'rifle']
    human_classes = ['criminal', 'person', 'Person', 'victim']
    combined_data_path = f'{dataset_path}-combined'
    data_dirs = os.listdir(dataset_path)
    for data_dir in data_dirs:
        data_path = os.path.join(dataset_path, data_dir)
        meta_file = os.path.join(data_path, 'data.yaml')

        weaponry = False
        person = False

        with open(meta_file, 'r') as f:
            classes = yaml.safe_load(f)['names']
        for cls in classes:
            if cls in weapon_classes:
                weaponry = True
            if cls in human_classes:
                person = True

        images_path = os.path.join(data_path, 'images')
        labels_path = os.path.join(data_path, 'labels')

        dest_dirs = {
            directory: os.path.join(combined_data_path, directory)
            for flag, directory in zip([person, weaponry], ["person", "weaponry"])
            if flag
        }
        class_wise_destinations = {
            key: {
                sub_dir: os.path.join(dest_dirs[key], sub_dir)
                for sub_dir in "images labels".split()
            }
            for key in dest_dirs.keys()
        }

        for class_destination in class_wise_destinations:
            for sub_dest in class_wise_destinations[class_destination]:
                sub_dest = class_wise_destinations[class_destination][sub_dest]
                if not os.path.exists(sub_dest):
                    print(f"Creating {sub_dest}")
                    os.makedirs(sub_dest, exist_ok=True)

        for label_file in os.listdir(labels_path):
            file_name = os.path.splitext(label_file)[0]
            label_path = os.path.join(labels_path, label_file)

            image_path = os.path.join(images_path, f"{file_name}.jpg")

            belong_h = False
            belong_w = False

            dest_labels = {
                "weapon": [],
                "person": [],
            }
            annotations = None

            with open(label_path, 'r') as original_annotation:
                annotations = original_annotation.readlines()

            for annotation in annotations:
                annotation_list = annotation.split()
                cls_id = int(annotation_list[0])
                class_name = classes[cls_id]
                annotation_list[0] = str(0)
                annotation = " ".join(annotation_list)
                if class_name in weapon_classes:
                    belong_w = True
                    dest_labels['weapon'].append(annotation)
                if class_name in human_classes:
                    belong_h = True
                    dest_labels['person'].append(annotation)
            if belong_h:
                image_destination = class_wise_destinations["person"]["images"]
                label_path = class_wise_destinations["person"]["labels"]
                label_destination = os.path.join(label_path, label_file)
                shutil.copy(image_path, image_destination)
                annotations = '\n'.join(dest_labels['person'])
                with open(label_destination, 'w') as final_annotation:
                    final_annotation.write(annotations)
            if belong_w:
                image_destination = class_wise_destinations["weaponry"]["images"]
                label_path = class_wise_destinations["weaponry"]["labels"]
                label_destination = os.path.join(label_path, label_file)
                shutil.copy(image_path, image_destination)
                annotations = '\n'.join(dest_labels['weapon'])
                with open(label_destination, 'w') as final_annotation:
                    final_annotation.write(annotations)
