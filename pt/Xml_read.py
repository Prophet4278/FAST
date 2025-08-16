import os
import xml.etree.ElementTree as ET

def parse_xml_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image information
    image_info = {
        "filename": root.find("filename").text,
        "size": {
            "width": int(root.find("size/width").text),
            "height": int(root.find("size/height").text),
            "depth": int(root.find("size/depth").text),
        },
    }

    # Extract object annotations
    annotations = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = {
            "xmin": int(obj.find("bndbox/xmin").text),
            "ymin": int(obj.find("bndbox/ymin").text),
            "xmax": int(obj.find("bndbox/xmax").text),
            "ymax": int(obj.find("bndbox/ymax").text),
        }

        annotations.append({
            "label": label,
            "bbox": bbox,
        })

    return image_info, annotations

def get_path(data,base_path):
    for i in data:
        path = i['file_name']
        directory,file_name=os.path.spilt(path)
        file_name_without_extension,extension = os.path.splitext(file_name)
        xml_path = os.path.join(base_path,file_name_without_extension+'.xml')
    return xml_path

def cat_ss(data,annotations):
    for i in data:
        i.update(annotations)
    datas = []
    datas = datas.extend(i)
    return datas

