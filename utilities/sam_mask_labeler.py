import numpy as np
import torch
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
import os
import sys
import warnings

warnings.filterwarnings('ignore')

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)


class SAMLabelerWithPolygon:
    def __init__(self, sam_predictor, image):
        self.predictor = sam_predictor
        self.image = image
        self.points = []
        self.labels = []
        self.boxes = []
        self.current_mask = None
        self.refined_mask = None
        self.current_box = None
        self.drawing_box = False
        self.box_start = None

        # Mask overlay tracking
        self.mask_overlay = None
        self.mask_visible = True

        # Polygon refinement state
        self.polygon_mode = False
        self.polygon_add_mode = True  # True for add, False for subtract
        self.current_polygon = []
        self.polygon_patches = []
        self.drawing_polygon = False

        # Set up the plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.imshow(self.image)
        self.update_title()

        # Connect mouse and keyboard events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Add control buttons
        self.setup_buttons()

        # Encode image
        self.predictor.set_image(self.image)

    def update_title(self):
        """Update title based on current mode"""
        if self.polygon_mode:
            mode_text = "POLYGON MODE - "
            if self.polygon_add_mode:
                mode_text += "ADD (green)"
            else:
                mode_text += "SUBTRACT (red)"
            mode_text += " | Click to draw polygon, Double-click to close"
        else:
            mode_text = "POINT/BOX MODE | Left: Positive, Right: Negative, Middle: Box"

        title = f"{mode_text}\nSpace: Generate, Enter: Save, P: Polygon mode, R: Reset"
        self.ax.set_title(title)

    def setup_buttons(self):
        """Setup control buttons"""
        # Button axes - arranged in two rows
        ax_generate = plt.axes([0.02, 0.02, 0.07, 0.04])
        ax_save = plt.axes([0.11, 0.02, 0.07, 0.04])
        ax_reset = plt.axes([0.20, 0.02, 0.07, 0.04])
        ax_undo = plt.axes([0.29, 0.02, 0.07, 0.04])

        ax_polygon = plt.axes([0.40, 0.02, 0.07, 0.04])
        ax_poly_add = plt.axes([0.49, 0.02, 0.07, 0.04])
        ax_poly_sub = plt.axes([0.58, 0.02, 0.07, 0.04])
        ax_preview = plt.axes([0.67, 0.02, 0.07, 0.04])
        ax_apply_poly = plt.axes([0.76, 0.02, 0.07, 0.04])
        ax_toggle_mask = plt.axes([0.85, 0.02, 0.07, 0.04])

        # Create buttons
        self.btn_generate = Button(ax_generate, 'Generate')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_undo = Button(ax_undo, 'Undo')

        self.btn_polygon = Button(ax_polygon, 'Polygon')
        self.btn_poly_add = Button(ax_poly_add, 'Add')
        self.btn_poly_sub = Button(ax_poly_sub, 'Subtract')
        self.btn_preview = Button(ax_preview, 'Preview')
        self.btn_apply_poly = Button(ax_apply_poly, 'Apply')
        self.btn_toggle_mask = Button(ax_toggle_mask, 'Hide/Show')

        # Connect button events
        self.btn_generate.on_clicked(self.generate_mask)
        self.btn_save.on_clicked(self.save_mask)
        self.btn_reset.on_clicked(self.reset_all)
        self.btn_undo.on_clicked(self.undo_last)

        self.btn_polygon.on_clicked(self.toggle_polygon_mode)
        self.btn_poly_add.on_clicked(self.set_polygon_add)
        self.btn_poly_sub.on_clicked(self.set_polygon_subtract)
        self.btn_preview.on_clicked(self.preview_polygon_effects)
        self.btn_apply_poly.on_clicked(self.apply_polygon)

    def preview_polygon_effects(self, event):
        """Preview what polygon effects would do without applying them"""
        if self.current_mask is None:
            print("Generate a mask first before previewing polygon effects")
            return

        if not self.polygon_patches:
            print("No polygons to preview")
            return

        print(f"\n=== POLYGON PREVIEW ===")

        # Start with current mask (or refined mask if it exists)
        preview_mask = self.refined_mask.copy() if self.refined_mask is not None else self.current_mask.copy()
        original_pixels = np.sum(preview_mask)
        print(f"Starting mask: {original_pixels} pixels")

        # Apply each polygon (preview only)
        for i, (patch, is_add_mode, polygon_points) in enumerate(self.polygon_patches):
            polygon_mask = self.create_polygon_mask(polygon_points)
            polygon_pixels = np.sum(polygon_mask)

            if polygon_pixels == 0:
                print(f"Polygon {i + 1}: EMPTY (no effect)")
                continue

            before_pixels = np.sum(preview_mask)

            if is_add_mode:
                preview_mask_after = np.logical_or(preview_mask, polygon_mask)
            else:
                preview_mask_after = np.logical_and(preview_mask, ~polygon_mask)

            after_pixels = np.sum(preview_mask_after)
            change = after_pixels - before_pixels

            mode_text = 'ADD' if is_add_mode else 'SUBTRACT'
            print(f"Polygon {i + 1} ({mode_text}): {polygon_pixels} polygon pixels → {change:+d} mask change")

            preview_mask = preview_mask_after

        final_pixels = np.sum(preview_mask)
        total_change = final_pixels - original_pixels
        print(f"Final result: {original_pixels} → {final_pixels} pixels ({total_change:+d} total change)")
        print("=== END PREVIEW ===\n")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.polygon_mode:
            self.handle_polygon_click(x, y, event)
        else:
            self.handle_normal_click(x, y, event)

    def handle_normal_click(self, x, y, event):
        """Handle clicks in normal point/box mode"""
        if event.button == 1:  # Left click - positive point
            self.points.append([x, y])
            self.labels.append(1)
            self.ax.plot(x, y, 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
            print(f"Added positive point at ({x}, {y})")

        elif event.button == 3:  # Right click - negative point
            self.points.append([x, y])
            self.labels.append(0)
            self.ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
            print(f"Added negative point at ({x}, {y})")

        elif event.button == 2:  # Middle click - start box
            self.drawing_box = True
            self.box_start = (x, y)
            print("Started drawing box...")

        self.fig.canvas.draw()

    def handle_polygon_click(self, x, y, event):
        """Handle clicks in polygon mode"""
        if event.button == 1:  # Left click in polygon mode
            if event.dblclick:
                # Double click - close polygon
                self.close_polygon()
            else:
                # Single click - add point to polygon
                self.current_polygon.append([x, y])

                # Draw point
                color = 'g' if self.polygon_add_mode else 'r'
                self.ax.plot(x, y, f'{color}o', markersize=6, markeredgecolor='white', markeredgewidth=1)

                # Draw line to previous point
                if len(self.current_polygon) > 1:
                    prev_point = self.current_polygon[-2]
                    self.ax.plot([prev_point[0], x], [prev_point[1], y], f'{color}-', linewidth=2)

                print(f"Added polygon point at ({x}, {y})")
                self.fig.canvas.draw()

        elif event.button == 3:  # Right click - cancel current polygon
            self.cancel_polygon()

    def close_polygon(self):
        """Close the current polygon"""
        if len(self.current_polygon) < 3:
            print("Need at least 3 points to close polygon")
            return

        print(f"Closing polygon with {len(self.current_polygon)} points")
        print(f"Points: {self.current_polygon}")

        # Test polygon mask creation
        test_mask = self.create_polygon_mask(self.current_polygon)
        test_pixels = np.sum(test_mask)
        print(f"Polygon will affect {test_pixels} pixels")

        if test_pixels == 0:
            print("Warning: Polygon has no area! Check if points are valid.")
            return

        # Close the polygon visually
        first_point = self.current_polygon[0]
        last_point = self.current_polygon[-1]
        color = 'g' if self.polygon_add_mode else 'r'
        self.ax.plot([last_point[0], first_point[0]], [last_point[1], first_point[1]], f'{color}-', linewidth=2)

        # Create polygon patch
        poly_patch = Polygon(self.current_polygon,
                             facecolor=color,
                             alpha=0.3,
                             edgecolor=color,
                             linewidth=2)
        self.ax.add_patch(poly_patch)
        self.polygon_patches.append((poly_patch, self.polygon_add_mode, self.current_polygon.copy()))

        mode_text = 'ADD' if self.polygon_add_mode else 'SUBTRACT'
        print(f"Created {mode_text} polygon with {test_pixels} pixels")

        # Reset for next polygon
        self.current_polygon = []
        self.fig.canvas.draw()

    def cancel_polygon(self):
        """Cancel the current polygon being drawn"""
        if self.current_polygon:
            print("Cancelled current polygon")
            self.current_polygon = []
            # Redraw everything to remove partial polygon
            self.redraw_display()

    def apply_polygon(self, event):
        """Apply all drawn polygons to refine the mask"""
        if self.current_mask is None:
            print("Generate a mask first before applying polygon refinements")
            return

        if not self.polygon_patches:
            print("No polygons to apply")
            return

        print(f"Applying {len(self.polygon_patches)} polygon refinements...")

        # Start with current mask (or refined mask if it exists)
        if self.refined_mask is None:
            self.refined_mask = self.current_mask.copy()
            print(f"Starting with original mask ({np.sum(self.current_mask)} pixels)")
        else:
            print(f"Starting with refined mask ({np.sum(self.refined_mask)} pixels)")

        # Apply each polygon
        for i, (patch, is_add_mode, polygon_points) in enumerate(self.polygon_patches):
            polygon_mask = self.create_polygon_mask(polygon_points)
            polygon_pixels = np.sum(polygon_mask)

            print(
                f"  Polygon {i + 1}: {len(polygon_points)} points, {polygon_pixels} pixels, {'ADD' if is_add_mode else 'SUBTRACT'}")

            if polygon_pixels == 0:
                print(f"    Warning: Polygon {i + 1} has no pixels, skipping")
                continue

            before_pixels = np.sum(self.refined_mask)

            if is_add_mode:
                # Add polygon area to mask
                self.refined_mask = np.logical_or(self.refined_mask, polygon_mask)
            else:
                # Subtract polygon area from mask
                self.refined_mask = np.logical_and(self.refined_mask, ~polygon_mask)

            after_pixels = np.sum(self.refined_mask)
            change = after_pixels - before_pixels
            print(f"    Result: {before_pixels} → {after_pixels} pixels ({change:+d})")

        # Update display
        self.show_mask(self.refined_mask)
        print(f"Final refined mask: {np.sum(self.refined_mask)} pixels")

        # Clear polygon patches
        for patch, _, _ in self.polygon_patches:
            patch.remove()
        self.polygon_patches = []
        self.fig.canvas.draw()

    def create_polygon_mask(self, polygon_points):
        """Create a binary mask from polygon points"""
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Convert to integer coordinates and ensure proper format
        polygon_array = np.array(polygon_points, dtype=np.int32)

        # Ensure we have at least 3 points
        if len(polygon_array) < 3:
            print(f"Warning: Polygon has only {len(polygon_array)} points, need at least 3")
            return mask.astype(bool)

        # Fill polygon - cv2.fillPoly expects [array_of_points]
        cv2.fillPoly(mask, [polygon_array], 255)

        # Convert to boolean
        return mask > 0

    def toggle_polygon_mode(self, event):
        """Toggle between polygon and normal mode"""
        self.polygon_mode = not self.polygon_mode
        self.current_polygon = []
        self.update_title()
        print(f"Polygon mode: {'ON' if self.polygon_mode else 'OFF'}")

    def set_polygon_add(self, event):
        """Set polygon mode to add"""
        self.polygon_add_mode = True
        self.update_title()
        print("Polygon mode: ADD")

    def set_polygon_subtract(self, event):
        """Set polygon mode to subtract"""
        self.polygon_add_mode = False
        self.update_title()
        print("Polygon mode: SUBTRACT")

    def on_release(self, event):
        if event.inaxes != self.ax or not self.drawing_box or self.polygon_mode:
            return

        if event.button == 2:  # Middle click release - end box
            x, y = int(event.xdata), int(event.ydata)
            if self.box_start:
                x1, y1 = self.box_start
                box = [min(x1, x), min(y1, y), max(x1, x), max(y1, y)]
                self.boxes = [box]  # Replace any existing box

                # Draw box
                if self.current_box:
                    self.current_box.remove()

                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='yellow', facecolor='none'
                )
                self.current_box = self.ax.add_patch(rect)
                print(f"Added box: {box}")

            self.drawing_box = False
            self.box_start = None
            self.fig.canvas.draw()

    def on_motion(self, event):
        if not self.drawing_box or not self.box_start or event.inaxes != self.ax or self.polygon_mode:
            return

        # Update box preview while drawing
        x, y = int(event.xdata), int(event.ydata)
        x1, y1 = self.box_start

        if self.current_box:
            self.current_box.remove()

        rect = patches.Rectangle(
            (min(x1, x), min(y1, y)), abs(x - x1), abs(y - y1),
            linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.5
        )
        self.current_box = self.ax.add_patch(rect)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == ' ':  # Space - generate mask
            self.generate_mask(None)
        elif event.key == 'enter':  # Enter - save mask
            self.save_mask(None)
        elif event.key == 'r':  # R - reset
            self.reset_all(None)
        elif event.key == 'u':  # U - undo
            self.undo_last(None)
        elif event.key == 'p':  # P - toggle polygon mode
            self.toggle_polygon_mode(None)
        elif event.key == 'v':  # V - toggle mask visibility
            self.toggle_mask_visibility(None)
        elif event.key == 't':  # T - preview polygon effects
            self.preview_polygon_effects(None)
        elif event.key == 'a' and self.polygon_mode:  # A - polygon add mode
            self.set_polygon_add(None)
        elif event.key == 's' and self.polygon_mode:  # S - polygon subtract mode
            self.set_polygon_subtract(None)
        elif event.key == 'escape' and self.polygon_mode:  # ESC - cancel polygon
            self.cancel_polygon()

    def generate_mask(self, event):
        """Generate mask using current points and boxes"""
        if not self.points and not self.boxes:
            print("Please add at least one point or box before generating mask")
            return

        try:
            # Prepare inputs
            input_points = np.array(self.points) if self.points else None
            input_labels = np.array(self.labels) if self.labels else None
            input_boxes = np.array(self.boxes) if self.boxes else None

            # Generate mask
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_boxes,
                multimask_output=True
            )

            # Select best mask
            best_idx = np.argmax(scores)
            self.current_mask = masks[best_idx]
            self.refined_mask = None  # Reset refined mask

            # Display mask
            self.show_mask(self.current_mask)
            print(f"Generated mask with score: {scores[best_idx]:.3f}")

        except Exception as e:
            print(f"Error generating mask: {e}")

    def toggle_mask_visibility(self, event):
        """Toggle mask visibility on/off"""
        self.mask_visible = not self.mask_visible

        if self.mask_visible:
            # Show mask if we have one
            mask_to_show = self.refined_mask if self.refined_mask is not None else self.current_mask
            if mask_to_show is not None:
                self.show_mask(mask_to_show)
            print("Mask visibility: ON")
        else:
            # Hide mask
            if self.mask_overlay is not None:
                self.mask_overlay.remove()
                self.mask_overlay = None
                self.fig.canvas.draw()
            print("Mask visibility: OFF")

    def show_mask(self, mask, alpha=0.6):
        """Display mask overlay, replacing any previous mask"""
        # Remove previous mask overlay if it exists
        if self.mask_overlay is not None:
            self.mask_overlay.remove()
            self.mask_overlay = None

        # Only show mask if visibility is enabled
        if self.mask_visible:
            # Create colored mask
            color_mask = np.zeros((*mask.shape, 4))
            color_mask[mask] = [0, 1, 0, alpha]  # Green with transparency

            # Display new mask and store reference
            self.mask_overlay = self.ax.imshow(color_mask)

        self.fig.canvas.draw()
        print(f"Updated mask display (pixels: {np.sum(mask)}, visible: {self.mask_visible})")

    def redraw_display(self):
        """Redraw the entire display"""
        self.ax.clear()
        self.mask_overlay = None  # Reset mask overlay reference
        self.ax.imshow(self.image)
        self.update_title()

        # Redraw points
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            color = 'go' if label == 1 else 'ro'
            self.ax.plot(point[0], point[1], color, markersize=8, markeredgecolor='white', markeredgewidth=2)

        # Redraw box
        if self.boxes and self.current_box is None:
            box = self.boxes[0]
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            self.current_box = self.ax.add_patch(rect)

        # Redraw mask if it exists
        mask_to_show = self.refined_mask if self.refined_mask is not None else self.current_mask
        if mask_to_show is not None:
            self.show_mask(mask_to_show)

        # Redraw polygon patches
        for patch, is_add_mode, _ in self.polygon_patches:
            self.ax.add_patch(patch)

        self.fig.canvas.draw()

    def save_mask(self, event):
        """Save the current mask as binary (black background, white mask)"""
        mask_to_save = self.refined_mask if self.refined_mask is not None else self.current_mask

        if mask_to_save is None:
            print("No mask to save. Generate a mask first.")
            return

        output_path = getattr(self, 'output_path', 'mask_output.png')

        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Save as binary mask (white = object, black = background)
            mask_img = (mask_to_save * 255).astype(np.uint8)
            if len(mask_img.shape) > 2:
                mask_img = mask_img.squeeze()

            success = cv2.imwrite(output_path, mask_img)
            if success:
                print(f"Mask saved to: {output_path}")
                print(f"Mask contains {np.sum(mask_to_save)} pixels")
            else:
                raise cv2.error("cv2.imwrite returned False")

        except Exception as e:
            print(f"Error saving mask: {e}")
            fallback_name = os.path.abspath("mask_output.png")
            try:
                cv2.imwrite(fallback_name, mask_img)
                print(f"Saved to fallback location: {fallback_name}")
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")

    def reset_all(self, event):
        """Reset all points, boxes, masks, and polygons"""
        self.points = []
        self.labels = []
        self.boxes = []
        self.current_mask = None
        self.refined_mask = None
        self.current_polygon = []
        self.mask_overlay = None  # Reset mask overlay reference

        # Clear polygon patches
        for patch, _, _ in self.polygon_patches:
            patch.remove()
        self.polygon_patches = []

        # Clear plot
        self.ax.clear()
        self.ax.imshow(self.image)
        self.update_title()

        if self.current_box:
            self.current_box = None

        self.fig.canvas.draw()
        print("Reset all annotations")

    def undo_last(self, event):
        """Undo last action"""
        if self.current_polygon:
            # Remove last point from current polygon
            self.current_polygon.pop()
            self.redraw_display()
            print("Removed last polygon point")
        elif self.polygon_patches:
            # Remove last polygon
            patch, _, _ = self.polygon_patches.pop()
            patch.remove()
            self.fig.canvas.draw()
            print("Removed last polygon")
        elif self.boxes:
            self.boxes.pop()
            if self.current_box:
                self.current_box.remove()
                self.current_box = None
            print("Removed last box")
        elif self.points:
            self.points.pop()
            self.labels.pop()
            self.redraw_display()
            print("Removed last point")
        else:
            print("Nothing to undo")


SAM_FILENAME_MAP = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


def get_arguments():
    parser = argparse.ArgumentParser(description='Interactive SAM-based mask labeling tool with polygon refinement')

    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for mask (default: <input_dir>/<image_name>_mask.png)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input image)')
    parser.add_argument('--suffix', type=str, default='_mask',
                        help='Suffix for output filename (default: _mask)')

    parser.add_argument('--sam_type', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, default=None,
                        help='Path to SAM checkpoint file (resolved from cache_dir if not provided)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for models (used to resolve SAM checkpoint)')

    return parser.parse_args()


def resolve_sam_checkpoint(args):
    """Resolve SAM checkpoint from cache_dir if not explicitly provided."""
    if args.sam_checkpoint is not None:
        return

    if args.cache_dir is None:
        print("ERROR: Either --sam_checkpoint or --cache_dir must be provided.")
        sys.exit(1)

    filename = SAM_FILENAME_MAP.get(args.sam_type)
    if filename is None:
        print("ERROR: No known checkpoint filename for SAM type '{}'.".format(args.sam_type))
        print("  Please provide --sam_checkpoint explicitly.")
        sys.exit(1)

    sam_ckpt = os.path.join(args.cache_dir, "checkpoints", "sam", filename)
    if not os.path.exists(sam_ckpt):
        print("ERROR: Could not resolve SAM checkpoint from cache_dir.")
        print("  Expected: {}".format(sam_ckpt))
        print("  Run: python utilities/download_models.py --cache_dir {} --sam".format(args.cache_dir))
        sys.exit(1)

    args.sam_checkpoint = sam_ckpt
    print("Resolved SAM checkpoint from cache_dir: {}".format(sam_ckpt))


def setup_output_path(args):
    """Setup output path based on arguments."""
    if args.output is not None:
        return

    output_dir = args.output_dir if args.output_dir is not None else os.path.dirname(os.path.abspath(args.image))

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(args.image))[0]
    args.output = os.path.join(output_dir, f"{image_name}{args.suffix}.png")


def main():
    args = get_arguments()

    # Resolve SAM checkpoint from cache_dir if needed
    resolve_sam_checkpoint(args)

    # Setup output path
    setup_output_path(args)

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM checkpoint not found: {args.sam_checkpoint}")
        return

    print("Loading SAM model...")

    # Initialize SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Loaded image: {args.image}")
    print(f"Image shape: {image.shape}")
    print(f"Output will be saved to: {args.output}")
    print("\n=== SAM LABELER WITH POLYGON REFINEMENT ===")
    print("\nBasic Controls:")
    print("- Left click: Add positive point (green)")
    print("- Right click: Add negative point (red)")
    print("- Middle click & drag: Draw bounding box (yellow)")
    print("- Space: Generate mask")
    print("- Enter: Save mask")
    print("- R: Reset all")
    print("- U: Undo last")
    print("- V: Toggle mask visibility")
    print("\nPolygon Refinement:")
    print("- P: Toggle polygon mode")
    print("- In polygon mode:")
    print("  - Left click: Add polygon points")
    print("  - Double-click: Close polygon")
    print("  - Right click: Cancel current polygon")
    print("  - A: Set add mode (green)")
    print("  - S: Set subtract mode (red)")
    print("  - T: Preview polygon effects")
    print("  - Apply button: Apply all polygons to mask")
    print("\nClose the window when done.")

    # Create labeler
    labeler = SAMLabelerWithPolygon(predictor, image)
    labeler.output_path = args.output

    # Show interactive plot
    plt.show()


if __name__ == "__main__":
    main()