#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np
import torch
import time
import argparse
from ultralytics import YOLO, YOLOWorld
from dataclasses import dataclass
from typing import List, Optional
import colorsys

# Mobile SAM imports (optional)
try:
    from mobile_sam import sam_model_registry, SamPredictor

    MOBILE_SAM_AVAILABLE = True
except ImportError:
    MOBILE_SAM_AVAILABLE = False
    print("Mobile SAM not available. Install with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")


@dataclass
class SegmentationResult:
    """Class to store segmentation results."""
    box: np.ndarray
    label: int
    label_text: str
    score: float
    mask: np.ndarray


class OptimizedSegmentationNode(Node):
    """Optimized ROS2 node for object detection and segmentation using YOLOv11-seg or YOLO-World + Mobile SAM."""

    def __init__(self,
                 model_path: str,
                 model_type: str = 'yolo11-seg',
                 sam_checkpoint: str = None,
                 use_cuda: bool = True,
                 confidence_threshold: float = 0.5,
                 max_detections: int = 20,
                 class_list: List[str] = None,
                 filter_classes: List[str] = None,
                 show_boxes: bool = True,
                 input_topic: str = '/camera/image_raw/compressed',
                 output_topic: str = '/segmentation/result_image',
                 input_size: int = 640,
                 use_compressed: bool = True,
                 skip_visualization: bool = False,
                 use_fp16: bool = True):
        """Initialize the node with model and parameters."""
        super().__init__('optimized_segmentation_node')

        # Declare parameters
        self.declare_parameter('confidence_threshold', confidence_threshold)
        self.declare_parameter('max_detections', max_detections)
        self.declare_parameter('input_topic', input_topic)
        self.declare_parameter('output_topic', output_topic)
        self.declare_parameter('model_type', model_type)
        self.declare_parameter('input_size', input_size)
        self.declare_parameter('use_compressed', use_compressed)
        self.declare_parameter('show_boxes', show_boxes)
        self.declare_parameter('skip_visualization', skip_visualization)
        self.declare_parameter('use_fp16', use_fp16)

        # Store parameters
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.max_detections = self.get_parameter('max_detections').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.model_type = self.get_parameter('model_type').value
        self.input_size = self.get_parameter('input_size').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.show_boxes = self.get_parameter('show_boxes').value
        self.skip_visualization = self.get_parameter('skip_visualization').value
        self.use_fp16 = self.get_parameter('use_fp16').value

        # Validate and optimize input size
        if self.input_size <= 0:
            raise ValueError(f"Input size must be positive, got {self.input_size}")

        # Round to nearest multiple of 32 for better performance
        self.input_size = ((self.input_size + 31) // 32) * 32
        if self.input_size != self.get_parameter('input_size').value:
            self.get_logger().info(f"Adjusted input size to {self.input_size} (multiple of 32)")

        # Store class list
        self.class_list = class_list or [
            'person', 'car', 'bicycle', 'truck', 'bus',
            'dog', 'cat', 'backpack', 'chair', 'bottle'
        ]

        # Store filter classes (if None, all classes are allowed)
        self.filter_classes = set(filter_classes) if filter_classes else None

        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'Model type: {self.model_type}')
        self.get_logger().info(f'Input size: {self.input_size}x{self.input_size}')
        self.get_logger().info(f'FP16 mode: {self.use_fp16 and self.device == "cuda"}')
        self.get_logger().info(f'Skip visualization: {self.skip_visualization}')

        if self.filter_classes:
            self.get_logger().info(f'Filtering classes: {sorted(self.filter_classes)}')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Performance optimization caches
        self.color_cache = {}
        self.last_sam_image_id = None

        # Initialize models based on type
        if self.model_type == 'yolo-world':
            if not MOBILE_SAM_AVAILABLE:
                raise RuntimeError("Mobile SAM is required for YOLO-World mode but not available")
            if sam_checkpoint is None:
                raise ValueError("SAM checkpoint path is required for YOLO-World mode")

            self.get_logger().info('Initializing YOLO-World + Mobile SAM...')
            self.detection_model = self._init_yolo_world_optimized(model_path)
            self.sam_predictor = self._init_mobile_sam(sam_checkpoint)
            self.segmentation_method = 'sam'
        else:  # yolo11-seg
            self.get_logger().info('Initializing YOLOv11 segmentation model...')
            self.detection_model = self._init_yolo_seg_optimized(model_path)
            self.sam_predictor = None
            self.segmentation_method = 'integrated'

        # Initialize subscriber based on image type
        if self.use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.input_topic,
                self.compressed_image_callback,
                10
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                self.input_topic,
                self.raw_image_callback,
                10
            )

        # Initialize publisher
        self.result_image_pub = self.create_publisher(
            Image,
            self.output_topic,
            10
        )

        # Performance metrics
        self.processing_times = []
        self.last_log_time = time.time()
        self.frame_count = 0

        self.get_logger().info(f'Node initialized. Subscribing to: {self.input_topic}')
        self.get_logger().info(f'Publishing results to: {self.output_topic}')

    def _init_yolo_world_optimized(self, model_path):
        """Initialize YOLO-World model with optimizations."""
        self.get_logger().info(f"Loading YOLO-World model from: {model_path}")

        model = YOLOWorld(model_path)
        model.set_classes(self.class_list)

        # Optimization settings
        model.overrides['conf'] = self.confidence_threshold
        model.overrides['imgsz'] = self.input_size
        model.overrides['max_det'] = self.max_detections

        if self.device == 'cuda' and self.use_fp16:
            model.overrides['half'] = True

        model.to(self.device)

        # Warmup
        self._warmup_model(model)

        self.get_logger().info("YOLO-World model initialized with optimizations")
        return model

    def _init_yolo_seg_optimized(self, model_path):
        """Initialize YOLOv11 segmentation model with optimizations."""
        self.get_logger().info(f"Loading YOLOv11 segmentation model from: {model_path}")

        model = YOLO(model_path)

        # Optimization settings
        model.overrides['conf'] = self.confidence_threshold
        model.overrides['imgsz'] = self.input_size
        model.overrides['max_det'] = self.max_detections

        # Enable FP16 for faster GPU inference
        if self.device == 'cuda' and self.use_fp16:
            model.overrides['half'] = True
            self.get_logger().info("Enabled FP16 inference")

        model.to(self.device)

        # Warmup
        self._warmup_model(model)

        self.get_logger().info("YOLOv11 segmentation model initialized with optimizations")
        return model

    def _warmup_model(self, model):
        """Warmup model for better initial performance."""
        if self.device == 'cuda':
            self.get_logger().info("Warming up model...")
            dummy_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            for _ in range(3):  # Run 3 warmup iterations
                model.predict(dummy_img, verbose=False, stream=True)
            torch.cuda.synchronize()  # Wait for warmup to complete
            self.get_logger().info("Model warmup complete")

    def _init_mobile_sam(self, checkpoint_path):
        """Initialize Mobile SAM model."""
        self.get_logger().info(f"Loading Mobile SAM from: {checkpoint_path}")

        model_type = "vit_t"
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        sam.eval()  # Set to evaluation mode

        sam_predictor = SamPredictor(sam)

        self.get_logger().info("Mobile SAM initialized successfully")
        return sam_predictor

    def should_filter_class(self, label_text):
        """Check if a class should be filtered out."""
        if self.filter_classes is None:
            return False
        return label_text not in self.filter_classes

    def segment_with_sam_optimized(self, image, boxes):
        """Optimized batch SAM segmentation."""
        # Set image only if different from last
        if self.last_sam_image_id != id(image):
            self.sam_predictor.set_image(image)
            self.last_sam_image_id = id(image)

        masks = []
        for box in boxes:
            x1, y1, x2, y2 = box
            box_xyxy = np.array([x1, y1, x2, y2])

            # Single mask output for speed
            mask_output, _, _ = self.sam_predictor.predict(
                box=box_xyxy,
                multimask_output=False
            )

            if mask_output is not None and len(mask_output) > 0:
                masks.append(mask_output[0])
            else:
                masks.append(np.zeros((image.shape[0], image.shape[1]), dtype=bool))

        return masks

    def run_inference_optimized(self, image):
        """Optimized inference pipeline."""
        # Convert once
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.segmentation_method == 'integrated':
            # YOLOv11-seg direct inference
            results = self.detection_model.predict(
                source=image_rgb,
                verbose=False,
                stream=True,  # Stream mode for lower memory
                agnostic_nms=True  # Faster NMS
            )

            return self._process_yolo_results_fast(results, image.shape[:2])
        else:
            # YOLO-World + SAM
            results = self.detection_model.predict(
                source=image_rgb,
                conf=self.confidence_threshold,
                imgsz=self.input_size,
                max_det=self.max_detections,
                verbose=False,
                stream=True,
                agnostic_nms=True
            )

            return self._process_yolo_world_results_fast(results, image_rgb)

    def _process_yolo_results_fast(self, results, original_shape):
        """Fast batch processing of YOLO segmentation results."""
        segmentation_results = []

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue

            if hasattr(r, 'masks') and r.masks is not None:
                # Batch process all detections
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                scores = r.boxes.conf.cpu().numpy()
                masks = r.masks.data.cpu().numpy()

                # Pre-filter by class to avoid unnecessary processing
                for i in range(min(len(classes), self.max_detections)):
                    label_text = r.names.get(classes[i], f"class_{classes[i]}")

                    if self.should_filter_class(label_text):
                        continue

                    # Resize mask only if necessary
                    mask = masks[i]
                    if mask.shape[:2] != original_shape:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (original_shape[1], original_shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        ) > 0.5  # Convert back to binary
                    else:
                        mask = mask.astype(bool)

                    segmentation_results.append(SegmentationResult(
                        box=boxes_xyxy[i],
                        label=classes[i],
                        label_text=label_text,
                        score=scores[i],
                        mask=mask
                    ))

        return segmentation_results

    def _process_yolo_world_results_fast(self, results, image_rgb):
        """Fast processing of YOLO-World + SAM results."""
        segmentation_results = []
        boxes_to_segment = []
        detection_info = []

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()

            for i in range(min(len(classes), self.max_detections)):
                label_text = self.class_list[classes[i]] if classes[i] < len(self.class_list) else f"class_{classes[i]}"

                if self.should_filter_class(label_text):
                    continue

                boxes_to_segment.append(boxes_xyxy[i])
                detection_info.append({
                    'label': classes[i],
                    'label_text': label_text,
                    'score': scores[i]
                })

        # Batch segment with SAM
        if boxes_to_segment:
            masks = self.segment_with_sam_optimized(image_rgb, boxes_to_segment)

            for i, (box, info, mask) in enumerate(zip(boxes_to_segment, detection_info, masks)):
                segmentation_results.append(SegmentationResult(
                    box=box,
                    label=info['label'],
                    label_text=info['label_text'],
                    score=info['score'],
                    mask=mask
                ))

        return segmentation_results

    def _get_color_for_class(self, class_id: int) -> tuple:
        """Generate consistent, distinguishable color for each class using golden ratio."""
        if class_id not in self.color_cache:
            # Use golden ratio for better color distribution (same as validator)
            hue = (class_id * 0.618033988749895) % 1.0
            saturation = 0.85
            value = 0.95

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Note: validator uses RGB order, convert to BGR for OpenCV
            color = tuple(int(c * 255) for c in reversed(rgb))
            self.color_cache[class_id] = color

        return self.color_cache[class_id]

    def calculate_mask_centroid(self, mask):
        """Calculate the centroid of a binary mask efficiently (matching validator)."""
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            return None
        cy, cx = mask_coords.mean(axis=0).astype(int)
        return (cx, cy)

    def visualize_results_fast(self, image, results):
        """Visualization matching the validator script style."""
        if not results:
            return image

        h, w = image.shape[:2]

        # Create visualization
        vis_image = image.copy()
        overlay = np.zeros_like(image)

        # Calculate adaptive sizes based on image dimensions
        font_scale = 0.6 * (max(h, w) / 640)
        font_scale = max(0.4, min(font_scale, 1.2))
        thickness = max(1, int(2 * (max(h, w) / 640)))
        box_thickness = max(2, int(3 * (max(h, w) / 640)))

        # Sort by mask area (largest first) for better visibility
        sorted_results = sorted(results,
                                key=lambda x: np.sum(x.mask) if x.mask is not None else 0,
                                reverse=True)

        for result in sorted_results:
            # Use consistent class color
            color = self._get_color_for_class(result.label)

            # Draw mask with consistent color
            if result.mask is not None:
                overlay[result.mask] = color

            # Draw box with class color
            if self.show_boxes:
                x1, y1, x2, y2 = result.box.astype(int)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, box_thickness)

            # Calculate centroid for label placement
            if result.mask is not None:
                mask_coords = np.argwhere(result.mask)
                if len(mask_coords) > 0:
                    cy, cx = mask_coords.mean(axis=0).astype(int)
                else:
                    x1, y1, x2, y2 = result.box.astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            else:
                x1, y1, x2, y2 = result.box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Label with confidence
            label = f"{result.label_text} {result.score:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Center text
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2

            # Keep text in bounds
            text_x = max(5, min(text_x, w - text_width - 5))
            text_y = max(text_height + 5, min(text_y, h - 5))

            # Draw text with black outline for visibility
            cv2.putText(vis_image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(vis_image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

        # Blend with transparency (0.6 image + 0.4 overlay)
        vis_image = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)

        # Add simple performance overlay (no black header)
        if not self.skip_visualization and len(self.processing_times) > 0:
            fps = 1.0 / np.mean(self.processing_times[-10:])
            info_font_scale = 0.6
            info_thickness = 2

            # Simple overlay in top-left corner with semi-transparent background
            info_text = f"FPS: {fps:.0f} | Detections: {len(results)}"
            (text_w, text_h), _ = cv2.getTextSize(
                info_text, cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, info_thickness
            )

            # Draw semi-transparent background for readability
            overlay_bg = vis_image.copy()
            cv2.rectangle(overlay_bg, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay_bg, 0.3, 0)

            # Draw text
            cv2.putText(vis_image, info_text, (10, text_h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, (0, 255, 0),
                        info_thickness, cv2.LINE_AA)

        return vis_image

    def generate_colors(self, num_colors):
        """Deprecated - now using _get_color_for_class for consistency."""
        # Keep for backward compatibility but not used
        pass





    def process_image_optimized(self, cv_image, header):
        """Optimized image processing pipeline."""
        start_time = time.time()

        try:
            # Run inference
            results = self.run_inference_optimized(cv_image)

            # Skip visualization if requested
            if not self.skip_visualization:
                # Visualize results
                result_image = self.visualize_results_fast(cv_image, results)

                # Publish result
                result_msg = self.cv_bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
                result_msg.header = header
                self.result_image_pub.publish(result_msg)

            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)

            # Periodic logging
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_log_time > 5.0:
                avg_time = np.mean(self.processing_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(
                    f'Performance: {avg_time * 1000:.1f}ms avg, {fps:.1f} FPS, '
                    f'{len(results)} detections, {self.frame_count} frames processed'
                )
                self.last_log_time = current_time

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def compressed_image_callback(self, compressed_msg):
        """Process incoming compressed image."""
        try:
            # Fast decompression
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is not None:
                self.process_image_optimized(cv_image, compressed_msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in compressed callback: {str(e)}')

    def raw_image_callback(self, image_msg):
        """Process incoming raw image."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            if cv_image is not None:
                self.process_image_optimized(cv_image, image_msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in raw callback: {str(e)}')


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimized Segmentation Node')

    # Model configuration
    parser.add_argument('--model', type=str, default="yolo11n-seg.pt",
                        help='Path to model file (YOLOv11-seg or YOLO-World)')
    parser.add_argument('--model-type', type=str, choices=['yolo11-seg', 'yolo-world'],
                        default='yolo11-seg', help='Type of model to use')
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                        help='Path to Mobile SAM checkpoint (required for YOLO-World mode)')

    # Detection parameters
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--max-detections', type=int, default=20,
                        help='Maximum number of detections to process')

    # Input size parameter
    parser.add_argument('--input-size', type=int, default=640,
                        help='Input image size for YOLO model (will be rounded to multiple of 32)')

    # Visualization parameters
    parser.add_argument('--no-boxes', action='store_true',
                        help='Disable bounding boxes')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization entirely for maximum speed')
    parser.add_argument('--filter-classes', type=str, nargs='+', default=None,
                        help='Only show detections for these classes')

    # Classes (for YOLO-World)
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['person', 'car', 'bicycle', 'truck', 'bus',
                                 'dog', 'cat', 'backpack', 'chair', 'bottle'],
                        help='Classes to detect (for YOLO-World)')

    # Topics
    parser.add_argument('--input-topic', type=str,
                        default='/camera/image_raw/compressed',
                        help='Input image topic')
    parser.add_argument('--output-topic', type=str,
                        default='/segmentation/result_image',
                        help='Output segmentation result topic')

    # Image type
    parser.add_argument('--raw', action='store_true',
                        help='Use raw image topic instead of compressed')

    # Performance options
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU processing')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable FP16 inference (use FP32)')

    # Initialize ROS
    rclpy.init(args=args)

    # Parse arguments
    parsed_args = parser.parse_args()

    # Validate arguments
    if parsed_args.model_type == 'yolo-world' and parsed_args.sam_checkpoint is None:
        parser.error("--sam-checkpoint is required when using --model-type yolo-world")

    # Create and run node
    node = OptimizedSegmentationNode(
        model_path=parsed_args.model,
        model_type=parsed_args.model_type,
        sam_checkpoint=parsed_args.sam_checkpoint,
        use_cuda=not parsed_args.cpu,
        confidence_threshold=parsed_args.confidence,
        max_detections=parsed_args.max_detections,
        class_list=parsed_args.classes,
        filter_classes=parsed_args.filter_classes,
        show_boxes=not parsed_args.no_boxes,
        input_topic=parsed_args.input_topic,
        output_topic=parsed_args.output_topic,
        input_size=parsed_args.input_size,
        use_compressed=not parsed_args.raw,
        skip_visualization=parsed_args.skip_viz,
        use_fp16=not parsed_args.no_fp16
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()