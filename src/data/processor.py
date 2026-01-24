import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def parse_voc_annotation(xml_file, images_path):
    """Parses a single VOC XML annotation file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(images_path, image_name)

    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append([image_path, label, xmin, ymin, xmax, ymax])
    return boxes

def process_dataset(annotations_path, images_path):
    """Processes all annotations in a directory and returns a DataFrame."""
    all_boxes = []
    xml_files = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]
    
    print(f"Parsing {len(xml_files)} annotations...")
    for xml_file in tqdm(xml_files):
        full_path = os.path.join(annotations_path, xml_file)
        all_boxes.extend(parse_voc_annotation(full_path, images_path))
        
    df = pd.DataFrame(all_boxes, columns=["image_path", "label", "xmin", "ymin", "xmax", "ymax"])
    print(f"✅ Parsed {len(df)} objects.")
    return df
