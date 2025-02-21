import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from .sam2.build_sam import build_sam2
from .sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import shutil
from .utils.general_utils import save_boxes

class MaskGenerator:
    def __init__(self, config, image_list, subfolders):
        self.config = config
        self.sam2_checkpoint = self.config['sam2_checkpoint']
        self.model_cfg = self.config['model_cfg']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.config['grounding_dino_config'], 
            model_checkpoint_path=self.config['dino_checkpoint'],
            device=self.device
        )

        self.text = self.config['text_object_prompt']

        if self.config['floor'] == True:
            self.floor_text = 'floor'
            self.text = f"{self.text}.{self.floor_text}"
        else:
            self.floor_text = ''
            

        self.image_list = image_list
        self.subfolders = subfolders
        self.output_folder_mask = 'masks/'
        self.output_folder_mask_colored = 'masks_colored/'
        self.output_folder_annotation = 'annotations/'
        self.output_folder_boxes = 'boxes/'

        print("> MaskGenerator initialized")

    def generate_masks(self):

        for i, subfolder in enumerate(self.subfolders):
            print(f"> Processing subfolder {subfolder}")

            if os.path.exists(os.path.join(subfolder, self.output_folder_mask)):
                shutil.rmtree(os.path.join(subfolder, self.output_folder_mask))
            os.makedirs(os.path.join(subfolder, self.output_folder_mask), exist_ok=True)
    
            if os.path.exists(os.path.join(subfolder, self.output_folder_mask_colored)):
                shutil.rmtree(os.path.join(subfolder, self.output_folder_mask_colored))
            os.makedirs(os.path.join(subfolder, self.output_folder_mask_colored), exist_ok=True)

            if os.path.exists(os.path.join(subfolder, self.output_folder_annotation)):
                shutil.rmtree(os.path.join(subfolder, self.output_folder_annotation))
            os.makedirs(os.path.join(subfolder, self.output_folder_annotation), exist_ok=True)
        
            if os.path.exists(os.path.join(subfolder, self.output_folder_boxes)):
                shutil.rmtree(os.path.join(subfolder, self.output_folder_boxes))
            os.makedirs(os.path.join(subfolder, self.output_folder_boxes), exist_ok=True)

            for idx, img_path in enumerate(tqdm(self.image_list[i])):
                #img_path = self.image_list[i][img]
                image_source, image = load_image(img_path)
                self.sam2_predictor.set_image(image_source)
                boxes, confidences, labels = predict(
                    model=self.grounding_model,
                    image=image,
                    caption=self.text,
                    box_threshold=self.config['box_threshold'],
                    text_threshold=self.config['text_threshold'],
                )

                # process the box prompt for SAM 2
                h, w, _ = image_source.shape
                boxes = boxes * torch.Tensor([w, h, w, h])
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                file_path = os.path.join(os.path.join(subfolder, self.output_folder_boxes), f'{(idx):04d}.txt')
                save_boxes(input_boxes, file_path)
                
                # FIXME: figure how does this influence the G-DINO model
                #torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

                if torch.cuda.get_device_properties(0).major >= 8:
                    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                if len(input_boxes) == 0:
                    print(f"No boxes found for {img_path}")
                    continue

                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                """
                Post-process the output of the model to get the masks, scores, and logits for visualization
                """
                # convert the shape to (n, H, W)
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                confidences = confidences.numpy().tolist()
                class_names = labels

                class_ids = np.array(list(range(len(class_names))))

                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(class_names, confidences)
                ]

                """
                Visualize image with supervision useful API
                """
                img = cv2.imread(img_path)
                detections = sv.Detections(
                    xyxy=input_boxes,  # (n, 4)
                    mask=masks.astype(bool),  # (n, h, w)
                    class_id=class_ids
                )

                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                cv2.imwrite(os.path.join(subfolder, os.path.join(self.output_folder_annotation, f'{(idx):04d}.png')), annotated_frame)

                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                cv2.imwrite(os.path.join(subfolder, os.path.join(self.output_folder_mask_colored, f'{(idx):04d}.png')), annotated_frame)
                
                mask_img = torch.zeros(masks.shape[1:], dtype=torch.uint8)  # Shape (H, W)

                for label, mask in enumerate(masks):
                    binary_mask = mask > 0
                    mask_img[binary_mask] = label + 1  # Assign label idx+1 to current mask
                mask_img_np = mask_img.cpu().numpy().astype(np.uint8)

                Image.fromarray(mask_img_np).save(os.path.join(subfolder, os.path.join(self.output_folder_mask, f'{(idx):04d}.png')), format='PNG')