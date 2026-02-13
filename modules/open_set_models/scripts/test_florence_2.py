import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description='Florence-2 Object Detection with Custom Classes')

    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output image (if not specified, displays instead)')
    parser.add_argument('--classes', '-c', nargs='+', required=True,
                        help='List of classes to detect (e.g., person car dog)')
    parser.add_argument('--model', '-m', type=str,
                        default='microsoft/Florence-2-large-ft',
                        help='Florence-2 model name (default: microsoft/Florence-2-large-ft)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache model files')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detected if not specified)')
    parser.add_argument('--max-tokens', type=int, default=1024,
                        help='Maximum new tokens to generate (default: 1024)')
    parser.add_argument('--num-beams', type=int, default=3,
                        help='Number of beams for generation (default: 3)')

    return parser.parse_args()


def load_model(model_name, cache_dir=None, device=None):
    """Load Florence-2 model and processor"""

    # Set cache directory
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir

    # Determine device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    if cache_dir:
        print(f"Cache directory: {cache_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    return model, processor, device, torch_dtype


def detect_with_classes(image_path, class_list, model, processor, device, torch_dtype,
                        max_tokens=1024, num_beams=3):
    """Detect objects using phrase grounding for EACH class separately"""

    print(f"\nProcessing image: {image_path}")
    print(f"Classes: {', '.join(class_list)}")

    image = Image.open(image_path).convert('RGB')

    all_bboxes = []
    all_labels = []

    # Process each class separately
    for class_name in class_list:
        print(f"  Detecting: {class_name}...")

        # Use phrase grounding for this specific class
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{class_name}"

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_tokens,
            num_beams=num_beams
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        result = processor.post_process_generation(
            generated_text,
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(image.width, image.height)
        )

        # Extract bboxes for this class
        grounding = result.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
        bboxes = grounding.get('bboxes', [])

        print(f"    Found {len(bboxes)} instances")

        # Add to combined results
        for bbox in bboxes:
            all_bboxes.append(bbox)
            all_labels.append(class_name)

    print(f"\nTotal detections: {len(all_bboxes)}")

    return {
        '<OPEN_VOCABULARY_DETECTION>': {
            'bboxes': all_bboxes,
            'bboxes_labels': all_labels,
            'polygons': [],
            'polygons_labels': []
        }
    }


def plot_results_cv2(image_path, result, output_path=None):
    """Plot with OpenCV"""

    image = cv2.imread(image_path)

    # DEBUG: Print the entire result structure
    print("\n=== DEBUG: Full Result ===")
    print(result)
    print("========================\n")

    # Get detections
    detections = result.get('<OPEN_VOCABULARY_DETECTION>', {})
    bboxes = detections.get('bboxes', [])
    labels = detections.get('bboxes_labels', [])

    print(f"Number of bboxes: {len(bboxes)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Bboxes: {bboxes}")
    print(f"Labels: {labels}")

    print(f"\nFound {len(bboxes)} detections")
    if labels:
        print(f"Detected labels: {set(labels)}")

    # Generate colors for each unique label
    np.random.seed(42)
    unique_labels = list(set(labels))
    colors = {label: tuple(np.random.randint(50, 255, 3).tolist())
              for label in unique_labels}

    # Draw detections
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[label]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label with background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save or display
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"\nSaved result to: {output_path}")
    else:
        cv2.imshow('Florence-2 Detections', image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def main():
    args = parse_args()

    # Load model
    model, processor, device, torch_dtype = load_model(
        args.model,
        args.cache_dir,
        args.device
    )

    # Run detection
    result = detect_with_classes(
        args.image,
        args.classes,
        model,
        processor,
        device,
        torch_dtype,
        args.max_tokens,
        args.num_beams
    )

    # Visualize results
    plot_results_cv2(args.image, result, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()