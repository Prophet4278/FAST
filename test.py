
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from pt import add_config
from pt.engine.trainer import PTrainer

# to register
from pt.modeling.meta_arch.rcnn import GuassianGeneralizedRCNN
from pt.modeling.proposal_generator.rpn import GuassianRPN
from pt.modeling.roi_heads.roi_heads import GuassianROIHead
import pt.data.datasets.builtin
from pt.modeling.backbone.vgg import build_vgg_backbone
from pt.modeling.anchor_generator import DifferentiableAnchorGenerator
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import PascalVOCDetectionEvaluator,COCOEvaluator
from detectron2.utils.logger import setup_logger

from pt.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from shutil import copyfile
import os

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from FAST_P import *

def image_inference(cfg, model, image_path):
    predictor = FASTPredictor(cfg)
    image = cv2.imread(image_path)
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = out.get_image()[:, :, ::-1]

    cv2.imwrite("output_image1.jpg", result_image)


def batch_inference(folder_path, cfg, output_folder):
    """
    Perform batch inference on images in the given folder and save the results.

    Args:
        folder_path (str): Path to the folder containing the images.
        cfg: Configuration object for the model.
        output_folder (str): Path to the folder where the output images will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the predictor
    predictor = FASTPredictor(cfg)

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image extensions
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            outputs = predictor(image)

            # Create a visualizer to draw the predictions on the image
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = out.get_image()[:, :, ::-1]

            # Save the output image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result_image)
            print(f"Saved: {output_path}")

# Example usage




def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    copyfile(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'))
    copyfile('pt/modeling/roi_heads/fast_rcnn.py', os.path.join(cfg.OUTPUT_DIR, 'fast_rcnn.py'))

    if cfg.UNSUPNET.Trainer == "pt":
        Trainer = PTrainer
    else:
        raise ValueError("Trainer Name is not found.")
      
    if args.eval_only:
        if cfg.UNSUPNET.Trainer in ["pt"]:
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            #cfg.MODEL.WEIGHTS = '/home/user/code/CMT-main/CMT_PT/output/c2f_0.02_s/model_final.pth'
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
            #res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            #image_path = '/home/user/datasets/CityScapes_FoggyCityScapes/JPEGImages/bremen_000035_000019_leftImg8bit_foggy_beta_0.02.jpg'
            #image_inference(cfg,ensem_ts_model.modelTeacher,image_path)

            input_folder = '/home/user/zhangdan/code/CMT-main/CMT_PT/data_test'
            output_folder = '/home/user/zhangdan/code/CMT-main/CMT_PT/data_test_out'
            batch_inference(input_folder, cfg, output_folder)

            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)



    
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


'''
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
import cv2
from detectron2.evaluation import pascal_voc_evaluation, inference_on_dataset
from detectron2.data import build_detection_test_loader

cfg = setup(args)
cfg.MODEL.WEIGHTS = "/home/user/code/CMT-main/CMT_PT/output/c2f_5/model_final.pth"
cfg.merge_from_file("/home/user/code/CMT-main/CMT_PT/configs/pt/test_c2f_5.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置检测的阈值
predictor = DefaultPredictor(cfg)
image = cv2.imread("/home/user/datasets/CityScapes_FoggyCityScapes/JPEGImages/bremen_000220_000019_leftImg8bit_foggy_beta_0.005.jpg")
outputs = predictor(image)
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Result", v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
evaluator = pascal_voc_evaluation.PascalVOCDetectionEvaluator("VOC2007_citytrain", cfg, False, output_dir="./testv_output/")
val_loader = build_detection_test_loader(cfg, "VOC2007_citytrain")
inference_on_dataset(predictor.model, val_loader, evaluator)
'''