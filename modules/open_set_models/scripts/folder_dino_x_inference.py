#!/usr/bin/env env python3
"""
DINO-X Batch Image Processor
Process multiple images with DINO-X Cloud API for object detection and segmentation
"""

import argparse
import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from pycocotools import mask as mask_utils
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch process images with DINO-X for object detection and segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-folder',
        '-i',
        type=str,
        required=True,
        help='Path to folder containing input images'
    )

    parser.add_argument(
        '--output-folder',
        '-o',
        type=str,
        default='./outputs/batch_detection',
        help='Path to folder for saving annotated images'
    )

    parser.add_argument(
        '--classes',
        '-c',
        type=str,
        required=True,
        help='Comma or dot-separated list of classes to detect (e.g., "car,person,dog" or "car . person . dog")'
    )

    parser.add_argument(
        '--confidence',
        '-conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (0.0 to 1.0)'
    )

    parser.add_argument(
        '--iou-threshold',
        '-iou',
        type=float,
        default=0.8,
        help='IoU threshold for NMS (0.0 to 1.0)'
    )

    parser.add_argument(
        '--api-token',
        '-t',
        type=str,
        default=None,
        help='API token for DDS Cloud API (can also be set via DDS_API_TOKEN env variable)'
    )

    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='DINO-X-1.0',
        help='Model name to use'
    )

    parser.add_argument(
        '--extensions',
        '-e',
        type=str,
        default='jpg,jpeg,png,bmp',
        help='Comma-separated list of image file extensions to process'
    )

    parser.add_argument(
        '--save-bbox-only',
        action='store_true',
        help='Save only bounding box annotations (no masks)'
    )

    return parser.parse_args()


def get_image_files(folder_path, extensions):
    """Get all image files from the folder with specified extensions"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f"*.{ext}"))
        image_files.extend(folder.glob(f"*.{ext.upper()}"))

    return sorted(image_files)


def process_image(client, img_path, text_prompt, confidence, iou_threshold, model_name):
    """Process a single image with DINO-X API"""
    # Convert image to base64
    image = image_to_base64(str(img_path))

    # Prepare API request
    api_path = "/v2/task/dinox/detection"
    api_body = {
        "model": model_name,
        "image": image,
        "prompt": {
            "type": "text",
            "text": text_prompt
        },
        "mask_format": "coco_rle",
        "targets": ["bbox", "mask"],
        "bbox_threshold": confidence,
        "iou_threshold": iou_threshold
    }

    # Run task
    task = V2Task(api_path=api_path, api_body=api_body)
    client.run_task(task)

    return task.result


def visualize_results(img_path, result, text_prompt, output_dir, save_bbox_only=False):
    """Visualize detection results and save annotated images"""
    objects = result["objects"]

    # Parse classes
    classes = [x.strip().lower() for x in text_prompt.replace(',', '.').split('.') if x.strip()]
    class_name_to_id = {name: id for id, name in enumerate(classes)}

    # Prepare detection data
    boxes = []
    masks = []
    confidences = []
    class_names = []
    class_ids = []

    for obj in objects:
        boxes.append(obj["bbox"])
        masks.append(mask_utils.decode(obj["mask"]))
        confidences.append(obj["score"])
        cls_name = obj["category"].lower().strip()
        class_names.append(cls_name)
        class_ids.append(class_name_to_id.get(cls_name, 0))

    if not boxes:
        print(f"  No detections found for {img_path.name}")
        return

    # Convert to numpy arrays
    boxes = np.array(boxes)
    masks = np.array(masks)
    class_ids = np.array(class_ids)

    # Create labels
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    # Load image
    img = cv2.imread(str(img_path))

    # Create detections
    detections = sv.Detections(
        xyxy=boxes,
        mask=masks.astype(bool),
        class_id=class_ids,
    )

    # Annotate with boxes
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    # Annotate with labels
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Save bbox-only version
    output_name = f"{img_path.stem}_annotated{img_path.suffix}"
    bbox_output_path = output_dir / output_name
    cv2.imwrite(str(bbox_output_path), annotated_frame)
    print(f"  Saved: {bbox_output_path}")

    # Annotate with masks and save
    if not save_bbox_only:
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        mask_output_name = f"{img_path.stem}_annotated_with_mask{img_path.suffix}"
        mask_output_path = output_dir / mask_output_name
        cv2.imwrite(str(mask_output_path), annotated_frame)
        print(f"  Saved: {mask_output_path}")


def main():
    """Main processing function"""
    args = parse_args()

    # Get API token
    api_token = args.api_token or os.getenv('DDS_API_TOKEN')
    if not api_token:
        raise ValueError(
            "API token not provided. Use --api-token argument or set DDS_API_TOKEN environment variable"
        )

    # Setup output directory
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(',')]

    # Get image files
    print(f"Searching for images in: {args.input_folder}")
    image_files = get_image_files(args.input_folder, extensions)

    if not image_files:
        print(f"No images found with extensions: {extensions}")
        return

    print(f"Found {len(image_files)} image(s) to process")
    print(f"Classes: {args.classes}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Output folder: {output_dir}")
    print("-" * 60)

    # Initialize API client
    config = Config(api_token)
    client = Client(config)

    # Format text prompt
    text_prompt = args.classes.replace(',', ' . ')

    # Process each image
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")

        try:
            # Process image
            result = process_image(
                client=client,
                img_path=img_path,
                text_prompt=text_prompt,
                confidence=args.confidence,
                iou_threshold=args.iou_threshold,
                model_name=args.model
            )

            # Visualize and save
            visualize_results(
                img_path=img_path,
                result=result,
                text_prompt=text_prompt,
                output_dir=output_dir,
                save_bbox_only=args.save_bbox_only
            )

        except Exception as e:
            print(f"  Error processing {img_path.name}: {str(e)}")
            continue

    print("-" * 60)
    print(f"Processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()