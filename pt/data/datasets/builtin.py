# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging
from detectron2.data.datasets.pascal_voc import register_pascal_voc

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
    "VOC2007_foggyval_coco": (
        "/home/user/datasets/CityScapes_FoggyCityScapes/JPEGImages",
        "/home/user/datasets/CityScapes_FoggyCityScapes/Annotations_1.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        
        #("VOC2007_citytrain", 'data/VOC2007_citytrain', "train", 8),
        #("VOC2007_foggytrain", 'data/VOC2007_foggytrain', "train", 8),
        #("VOC2007_foggyval", 'data/VOC2007_foggyval', "val", 8),
        #("VOC2007_foggytrain_0.02", 'data/VOC2007_foggytrain', "train_0.02", 8),
        #("VOC2007_foggyval_0.02", 'data/VOC2007_foggyval', "val_0.02", 8),
        #("VOC2007_citytrain1", 'data/VOC2007_citytrain1', "train", 1),
        #("VOC2007_cityval1", 'data/VOC2007_cityval1', "val", 1),
        #("VOC2007_bddtrain", 'data/VOC2007_bddtrain', "train", 8),
        #("VOC2007_bddval", 'data/VOC2007_bddval', "val", 8),
        #("VOC2007_kitti1", 'data/kitti', "train", 1),
        #("VOC2007_sim1", 'data/sim', "train", 1),
        
         #("VOC2007_citytrain", 'data/cityscape/VOC2007', "train_s", 8),#data/VOC2007_citytrain
        ("VOC2007_citytrain", '/home/user/zhangdan/datasets/cityscape/VOC2007', "train_s", 8),
        ("VOC2007_citytrain1", '/home/user/zhangdan/datasets/cityscape_caronly/VOC2007', "train_caronly1", 1),
        #("VOC2007_foggytrain_0.02", '/home/user/datasets/CityScapes_FoggyCityScapes', "train_foggy_0.02", 8),
        #("VOC2007_foggyval_0.02", '/home/user/datasets/CityScapes_FoggyCityScapes', "val_foggy_0.02", 8),
        #("VOC2007_citytrain_ada", '/home/user/datasets/cityscape_ada/VOC2007', "train_s", 8),
        #("VOC2007_citytrain_ada_mtkl", '/home/user/datasets/cityscape_ada/VOC2007', "train_mtkl", 8),
        #"VOC2007_citytrain_ada_ko", '/home/user/datasets/cityscape_ada/VOC2007', "train_ko", 8),
        #("VOC2007_citytrain_ada_af", '/home/user/datasets/cityscape_ada/VOC2007', "train_af", 8),
        # ("VOC2007_citytrain_ada_0.02_mtkl", '/home/user/datasets/cityscape_ada/VOC2007', "train_0.02_mtkl", 8),
        #("VOC2007_citytrain_ada_0.02_ao", '/home/user/datasets/cityscape_ada/VOC2007', "train_0.02_ao", 8),
        #("VOC2007_citytrain_ada_r", '/home/user/datasets/cityscape_ada/VOC2007', "train_sr", 8),
       #("VOC2007_citytrain_ada_0.02", '/home/user/datasets/cityscape_ada/VOC2007', "train_0.02_s", 8),
       #("VOC2007_citytrain_ada_0.02_2", '/home/user/datasets/cityscape_ada/VOC2007', "train_0.02_s2", 8),
        #("VOC2007_foggytrain", '/home/user/datasets/cityscape/VOC2007', "train_s", 8),#data/VOC2007_foggytrain
        ("VOC2007_foggytrain", '/home/user/zhangdan/datasets/CityScapes_FoggyCityScapes', "train_foggy", 8),
        #("VOC2007_foggyval", '/home/user/datasets/cityscape/VOC2007', "test_t", 8),#data/VOC2007_foggyval
        ("VOC2007_foggyval", '/home/user/zhangdan/datasets/CityScapes_FoggyCityScapes', "val_foggy", 8),
        #("VOC2007_citytrain1", 'data/cityscape/VOC2007', "train_s", 1),
        #("VOC2007_citytrain1", '/home/user/datasets/cityscape/VOC2007', "train_all", 1),
        ("VOC2007_cityval1", '/home/user/datasets/cityscape_caronly/VOC2007', "val_caronly", 1),
        # ("VOC2007_bddtrain", 'data/VOC2007_bddtrain', "train", 8),
        # ("VOC2007_bddval", 'data/VOC2007_bddval', "val", 8),
        #("VOC2007_kitti1", '/home/user/datasets/KITTI', "train_caronly", 1),
        #("VOC2007_kitti1_ada", '/home/user/datasets/KITTI_ada', "train_caronly_k2c", 1),
        #("VOC2007_kitti1_ada_af", '/home/user/datasets/KITTI_ada', "train_af", 1),
        ("VOC2007_kitti1_ada_mtkl", '/home/user/zhangdan/datasets/KITTI_ada', "train_caronly_mtkl_1", 1),
        #("VOC2007_kitti1_ada_kl", '/home/user/datasets/KITTI_ada', "train_k2c_kl", 1),
        #("VOC2007_sim1_ada", '/home/user/datasets/SIM_ADA/VOC2012', "train_sa", 1),
        #("VOC2007_sim1_ada_ko", '/home/user/datasets/SIM_ADA/VOC2012', "train_sko", 1),
        #("VOC2007_sim1_ada_af", '/home/user/datasets/SIM_ADA/VOC2012', "train_saf", 1),
        #("VOC2007_sim1_ada_hf", '/home/user/datasets/SIM_ADA/VOC2012', "train_shf", 1),
        #("VOC2007_sim1_ada_x", '/home/user/datasets/SIM_ADA/VOC2012', "train_sx", 1),
        #("VOC2007_sim1_ada_kl", '/home/user/datasets/SIM_ADA/VOC2012', "train_kl", 1),
        #("VOC2007_sim1_ada_mtkl", '/home/user/datasets/SIM_ADA/VOC2012', "train_mtkl", 1),
        #("Clipart1k_train",'/home/user/datasets/clipart',"trainval",20),
        #("Clipart1k_test",'/home/user/datasets/clipart',"test",20),
        #"VOC2007_sim1", '/home/user/datasets/SIM/VOC2012', "train_s", 1),
        #("VOC2012_train_ada",'/home/user/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',"train_ada",20),
        #("VOC2012_train_ada_ko",'/home/user/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',"train_ada_ko",20),
    ]
    for name, dirname, split, cls in SPLITS:
        year = 2012
        if cls == 1:
            class_names = ('car',)
        elif cls == 20:
            class_names = (
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            )
        elif cls == 8:
            class_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')
        elif cls == 7:
            class_names = ('truck', 'car', 'rider', 'person', 'motorcycle', 'bicycle', 'bus')
        else:
            raise RuntimeError
        register_pascal_voc(name, os.path.join(root, dirname), split, year, class_names)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


_root = os.getenv("DETECTRON2_DATASETS", "")
register_coco_unlabel(_root)
register_all_pascal_voc(_root)
