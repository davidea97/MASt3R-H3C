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
from .sam2.build_sam import build_sam2, build_sam2_video_predictor
from .sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import shutil
from .utils.general_utils import save_boxes
from .utils.video_utils import create_video_from_images


class MaskGenerator:
    def __init__(self, config, image_list, subfolders):
        self.config = config
        self.sam2_checkpoint = self.config['sam2_checkpoint']
        self.model_cfg = self.config['model_cfg']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.config['grounding_dino_config'], 
            model_checkpoint_path=self.config['dino_checkpoint'],
            device=self.device
        )


        self.floor_text = 'floor.'
        self.text = f"{self.floor_text}."

        self.image_list = image_list
        self.subfolders = subfolders
        self.output_folder_mask = 'masks/'
        self.output_folder_mask_colored = 'masks_colored/'
        self.output_folder_annotation = 'annotations/'

        print("> MaskGenerator initialized")

    def generate_masks(self):

        _, ext_example = os.path.splitext(self.image_list[0][0])
        for i, subfolder in enumerate(self.subfolders[:1]):
            print(f"> Processing subfolder {subfolder}")

            # Create the output_folders
            self.create_folder(subfolder)            
            list_len = len(self.image_list[i])
            inference_state = self.video_predictor.init_state(video_path=os.path.join(subfolder, "image"), list_len=list_len)

            # Let's process the first image to get the initial boxes
            init_objects = None
            input_boxes = None
            for idx, img_path in enumerate(tqdm(self.image_list[i])):
                
                image_source, image = load_image(img_path)
                self.sam2_predictor.set_image(image_source)
                boxes, confidences, labels = predict(
                    model=self.grounding_model,
                    image=image,
                    caption=self.text,
                    box_threshold=self.config['box_threshold'],
                    text_threshold=self.config['text_threshold'],
                )
                
                if idx == 0:
                    init_objects = labels
                    # init_boxes = boxes
                    h, w, _ = image_source.shape
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            for object_id, (label, box) in enumerate(zip(init_objects, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    box=box,
                )

            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            """
            Step 5: Visualize the segment results across the video and save them
            """

            save_dir = "./tracking_results"

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            ID_TO_OBJECTS = {i: obj for i, obj in enumerate(init_objects, start=1)}
            for frame_idx, segments in video_segments.items():
                img = cv2.imread(self.image_list[i][frame_idx])
                _, ext = os.path.splitext(self.image_list[i][frame_idx])
                image_name = os.path.basename(self.image_list[i][frame_idx]).split('.')[0]
                object_ids = list(segments.keys())
                masks = list(segments.values())
                masks = np.concatenate(masks, axis=0)
                
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                    mask=masks, # (n, h, w)
                    class_id=np.array(object_ids, dtype=np.int32),
                )

                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
                cv2.imwrite(os.path.join(subfolder, os.path.join(self.output_folder_annotation, f'{image_name}{ext}')), annotated_frame)
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                cv2.imwrite(os.path.join(subfolder, os.path.join(self.output_folder_mask_colored, f'{image_name}{ext}')), annotated_frame)

                mask_img = torch.zeros(masks.shape[1:], dtype=torch.uint8)  # Shape (H, W)

                for label, mask in enumerate(masks):
                    binary_mask = mask > 0
                    mask_img[binary_mask] = label + 1  # Assign label idx+1 to current mask
                mask_img_np = mask_img.cpu().numpy().astype(np.uint8)

                Image.fromarray(mask_img_np).save(os.path.join(subfolder, os.path.join(self.output_folder_mask, f'{image_name}{ext}')), format='PNG')

        return init_objects, ext_example
                

    def create_folder(self, subfolder):
        if os.path.exists(os.path.join(subfolder, self.output_folder_mask)):
            shutil.rmtree(os.path.join(subfolder, self.output_folder_mask))
        os.makedirs(os.path.join(subfolder, self.output_folder_mask), exist_ok=True)

        if os.path.exists(os.path.join(subfolder, self.output_folder_mask_colored)):
            shutil.rmtree(os.path.join(subfolder, self.output_folder_mask_colored))
        os.makedirs(os.path.join(subfolder, self.output_folder_mask_colored), exist_ok=True)

        if os.path.exists(os.path.join(subfolder, self.output_folder_annotation)):
            shutil.rmtree(os.path.join(subfolder, self.output_folder_annotation))
        os.makedirs(os.path.join(subfolder, self.output_folder_annotation), exist_ok=True)