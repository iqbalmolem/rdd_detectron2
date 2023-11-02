import os
import glob
import cv2

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from bbox_converter import yolobbox2bbox

def get_dataset(image_folder: str,
            label_folder: str) -> None:

  dataset = []
  id = 0

  # Exctract image information
  image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
  label_files = sorted(glob.glob(os.path.join(label_folder, '*.txt')))

  for image_file, label_file in zip(image_files, label_files):
    image = cv2.imread(image_file)
    height, width =  image[..., 0].shape[:2]

    annotations = []
    with open(label_file, 'r') as file:
      file_lines = file.readlines()
      for line in file_lines:
        line = line.strip().split()
        label = int(line[0])
        bbox = yolobbox2bbox(float(line[1]),float(line[2]), float(line[3]), float(line[4]), int(height), int(width))
        annotation = {
                    'bbox': bbox,
                    'bbox_mode': BoxMode(0),
                    'category_id': label
                }
        annotations.append(annotation)

      sample = {
            'file_name': image_file,
            'image_id': id,
            'height': int(height),
            'width': int(width),
            'annotations': annotations
        }

      dataset.append(sample)
      id += 1

  return dataset

def register_dataset():

    ls_class = ['0','1','2','3','4','5','6','7','8','9']

    DatasetCatalog.remove('rddtrain')
    DatasetCatalog.remove('rddtest')
    DatasetCatalog.remove('rddval')

    def train_dataset():
        return get_dataset('/content/yolo_merged_dataset/images/train','/content/yolo_merged_dataset/labels/train')

    def test_dataset():
        return get_dataset('/content/yolo_merged_dataset/images/test','/content/yolo_merged_dataset/labels/test')

    def val_dataset():
        return get_dataset('/content/yolo_merged_dataset/images/val','/content/yolo_merged_dataset/labels/val')

    DatasetCatalog.register('rddtrain', train_dataset)
    DatasetCatalog.register('rddtest', test_dataset)
    DatasetCatalog.register('rddval', val_dataset)

    MetadataCatalog.get("rddtrain").thing_classes = ls_class
    MetadataCatalog.get("rddtest").thing_classes = ls_class
    MetadataCatalog.get("rddval").thing_classes = ls_class