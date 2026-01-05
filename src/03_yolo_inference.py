"""
YOLO Inference

This script demonstrates how object detection works using a pretrained YOLO
model. We'll use YOLOv8 trained on the COCO dataset, which can detect 80
different object classes like people, cars, and birds. While it won't detect
solar panels (since they're not in COCO), this script teaches the mechanics of
how YOLO works, including bounding boxes, confidence scores, and non-maximum
suppression.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

def load_image(image_path):
    """
    Loads an image from disk and converts it to RGB color space.
    
    :param image_path: Path to the image file on disk.
    :return: The image as a numpy array in RGB format.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def load_pretrained_model():
    """
    Loads a pretrained YOLOv8 model trained on the COCO dataset.
    
    The COCO (Common Objects in Context) dataset contains 80 object classes
    including person, car, dog, chair, etc. When we call `YOLO('yolov8n.pt')`,
    the library downloads the pretrained weights automatically if they don't
    exist locally. We use the "nano" variant (`yolov8n`) because it's the
    smallest and fastest, which is perfect for learning and experimentation.
    
    The full size hierarchy is: nano < small < medium < large < xlarge, where
    larger models are more accurate but slower.
    
    :return: The loaded YOLO model object.
    """
    # The first time this runs, it will download the model weights (~6 MB)
    # and cache them locally
    model = YOLO('yolov8n.pt')
    
    return model

def run_inference(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Runs the YOLO model on an image to detect objects.
    
    When we call `model(image)`, a lot of things happen internally:
    1. The image is resized to the model's input size (typically 640x640)
    2. The image is normalized (pixel values scaled to [0, 1])
    3. The image is passed through the neural network
    4. The network outputs predictions for thousands of potential bounding boxes
    5. Low-confidence predictions are filtered out
    6. Non-maximum suppression (NMS) removes duplicate detections
    
    The result is a list of high-confidence, non-overlapping bounding boxes.
    
    :param model: The loaded YOLO model object.
    :param image: The RGB image array.
    :param conf_threshold: The minimum confidence score to keep a detection.
    :return: A Results object containing the detected objects.
    """
    # Run inference on the image
    # `conf` sets the confidence threshold (0.25 means we only keep detections
    # with confidence >= 25%)
    # The `verbose=False` parameter suppresses the progress bar and logging
    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    # YOLO always returns a list of results (one per image). Since we're only
    # processing a single image, we just need to take the first element.
    return results[0]

def analyze_detection_results(result):
    """    
    Prints a summary of the detected objects in the image.
    
    The result object contains the following attributes:
    - `boxes`: A list of bounding boxes (coordinates, confidence, class)
    - `names`: A dictionary mapping class IDs to class names (0: 'person', etc.)
    - `orig_shape`: The original image dimensions before resizing
    - `probs`: A list of class probabilities (for classification tasks)
    - `masks`: A list of segmentation masks (for instance segmentation)

    Each detection (bounding box) has:
    - `xyxy`: The bounding box coordinates in [x1, y1, x2, y2] format (top-left and bottom-right corners)
    - `conf`: The confidence score [0, 1] that the object is present
    - `cls`: The class ID of the detected object (0: 'person', 1: 'car', etc.)
    
    :param result: The YOLO results object.
    """    
    print(f"Image Dimensions: {result.orig_shape[0]}x{result.orig_shape[1]}")
    print(f"  → Original height: {result.orig_shape[0]} px")
    print(f"  → Original width:  {result.orig_shape[1]} px")

    boxes = result.boxes
    num_detections = len(boxes)
    print(f"\nDetections Found: {num_detections}")

    for i, box in enumerate(boxes):
        # Extract data from the box tensor
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        print(f"\nDetection #{i + 1}: [{label.upper()}]")
        print(f"  → Confidence: {confidence:.2%}")
        print(f"  → Class ID:   {cls_id}")
        print(f"  → Bounding Box (xyxy):")
        print(f"    - Top-Left:     ({coords[0]:.1f}, {coords[1]:.1f})")
        print(f"    - Bottom-Right: ({coords[2]:.1f}, {coords[3]:.1f})")
    
        # Calculate box dimensions
        width = coords[2] - coords[0]
        height = coords[3] - coords[1]
        print(f"    - Dimensions:   {width:.1f}x{height:.1f} pixels")

def parse_detections(results):
    """
    Extracts bounding boxes, confidence scores, and class IDs from the YOLO results object.
    
    :param results: The results object from YOLO inference.
    :return: A tuple of (boxes, confidences, class_ids, class_names) where:
             - boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] coordinates
             - confidences: numpy array of shape (N,) with confidence scores
             - class_ids: numpy array of shape (N,) with class indices
             - class_names: list of N class name strings
    """
    # Get the boxes tensor from the results
    boxes_data = results.boxes
    
    if boxes_data is None or len(boxes_data) == 0:
        return [], [], [], []
    
    # The `boxes` attribute contains all the bounding box information
    # `boxes.xyxy` gives us the coordinates in [x1, y1, x2, y2] format
    # `boxes.conf` gives us the confidence scores
    # `boxes.cls` gives us the class indices
    boxes = boxes_data.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
    confidences = boxes_data.conf.cpu().numpy()  # [conf1, conf2, ...]
    class_ids = boxes_data.cls.cpu().numpy().astype(int)  # [cls1, cls2, ...]
    
    # Get the class names from the model
    # The model has a `names` attribute that maps class IDs to class names
    class_names = [results.names[class_id] for class_id in class_ids]
    
    return boxes, confidences, class_ids, class_names

def draw_detections(image, boxes, confidences, class_names):
    """
    Draws bounding boxes and labels on the image.
    
    For each detection, we draw:
    1. A colored rectangle around the object
    2. A text label showing the class name and confidence score
    
    :param image: The RGB image array.
    :param boxes: numpy array of shape (N, 4) with bounding box coordinates.
    :param confidences: numpy array of shape (N,) with confidence scores.
    :param class_names: list of class name strings.
    :return: A copy of the image with bounding boxes and labels drawn around the detected objects.
    """
    # Make a copy so we don't modify the original
    img_with_boxes = image.copy()
    
    # Draw each detection
    for box, conf, class_name in zip(boxes, confidences, class_names):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw the bounding box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Create the label text
        label = f"{class_name}: {conf:.2f}"
        
        # Draw a filled rectangle for the text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1

        # Get the size of the text to create a background box
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Draw the background rectangle for the text
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw the text on top of the background
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness
        )
    
    return img_with_boxes

def visualize_results(image, results, output_path):
    """
    Creates a side-by-side visualization of the original image and the YOLO 
    detections and saves the result.
    
    :param image: The original RGB image array.
    :param results: The YOLO results object.
    :param output_path: Path where the figure will be saved.
    """
    # Parse the YOLO results
    boxes, confidences, class_ids, class_names = parse_detections(results)

    # Draw the detections
    img_with_boxes = draw_detections(image, boxes, confidences, class_names)
    
    # Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    
    axes[1].imshow(img_with_boxes)
    axes[1].set_title(f'YOLO Detections ({len(boxes)} objects)')
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
def visualize_nms(model, image, output_path, iou_threshold=0.45):
    """
    Creates a side-by-side visualization of the detections before and after 
    Non-Maximum Suppression (NMS) and saves the result.

    During inference, YOLO generates thousands of candidate bounding boxes at
    different positions and scales. Many of these boxes will overlap and point
    to the same object. For example, when detecting a person, we might get 10
    slightly different boxes all covering roughly the same area.
    
    Non-maximum suppression solves this by:
    1. Sorting all boxes by confidence score (highest first)
    2. Taking the highest-confidence box and keeping it
    3. Removing all other boxes that overlap significantly (IoU > threshold)
    4. Repeating with the next highest-confidence box
    
    IoU (Intersection over Union) measures how much two boxes overlap:
    IoU = (Area of Overlap) / (Area of Union)
    
    An IoU of 0.5 means the boxes overlap by 50%. The default NMS threshold is
    typically 0.45, meaning boxes with >45% overlap are considered duplicates.

    While this script visualizes the steps manually for educational purposes, 
    YOLO performs this NMS process internally by default during every inference call.

    :param model: The loaded YOLO model object.
    :param image: The RGB image array.
    :param output_path: Path where the figure will be saved.
    :param iou_threshold: The IoU threshold for NMS.
    """
    # 1. Get RAW results (bypass NMS by setting iou threshold to 1.0)
    # This shows all candidate boxes that passed the confidence threshold.
    raw_results = run_inference(model, image, iou_threshold=1.0)

    # 2. Get SUPPRESSED results (standard NMS)
    suppressed_results = run_inference(model, image, iou_threshold=iou_threshold)

    # Parse the results
    raw_boxes, raw_conf, _, raw_names = parse_detections(raw_results)
    sup_boxes, sup_conf, _, sup_names = parse_detections(suppressed_results)
    
    # Draw the detections
    img_raw = draw_detections(image, raw_boxes, raw_conf, raw_names)
    img_sup = draw_detections(image, sup_boxes, sup_conf, sup_names)

    # Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_raw)
    axes[0].set_title("Raw Detections (IoU=1.0)")
    axes[0].axis('off')
    
    axes[1].imshow(img_sup)
    axes[1].set_title(f"Suppressed Detections (IoU={iou_threshold:.2f})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate IoU between first two boxes as an example
    # box1 = boxes.xyxy[0].cpu().numpy()
    # box2 = boxes.xyxy[1].cpu().numpy()
    # iou = calculate_iou(box1, box2)

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    
    This function computes how much two boxes overlap. The IoU is calculated as
    the area of intersection divided by the area of union. An IoU of 1.0 means
    the boxes are identical, while 0.0 means they don't overlap at all.
    
    :param box1: First bounding box as [x1, y1, x2, y2].
    :param box2: Second bounding box as [x1, y1, x2, y2].
    :return: The IoU value between 0 and 1.
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def create_test_image_with_common_objects():
    """
    Creates a synthetic image with COCO-detectable objects.
    
    :return: A synthetic RGB image.
    """
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Background: Sky and Road
    image[:300, :] = [135, 206, 235]  # Sky (Sky Blue)
    image[300:, :] = [50, 50, 50]     # Road (Dark Grey)
    
    # 2. Draw "Cars" (Rectangles with "wheels")
    # Car 1
    cv2.rectangle(image, (50, 320), (150, 380), (0, 0, 255), -1)  # Red car body
    cv2.circle(image, (70, 385), 10, (0, 0, 0), -1)               # Wheel
    cv2.circle(image, (130, 385), 10, (0, 0, 0), -1)              # Wheel
    
    # Car 2
    cv2.rectangle(image, (200, 340), (320, 400), (255, 0, 0), -1) # Blue car body
    cv2.circle(image, (220, 405), 12, (0, 0, 0), -1)              # Wheel
    cv2.circle(image, (300, 405), 12, (0, 0, 0), -1)              # Wheel

    # A "Bus" (Larger rectangle)
    cv2.rectangle(image, (350, 250), (550, 380), (0, 255, 255), -1) # Yellow bus
    
    # 3. A "Person" (Stick figure-ish)
    cv2.circle(image, (580, 320), 10, (255, 224, 189), -1)        # Head
    cv2.line(image, (580, 330), (580, 360), (200, 0, 0), 3)       # Body
    
    # Traffic Light
    cv2.rectangle(image, (20, 50), (50, 150), (20, 20, 20), -1)   # Housing
    cv2.circle(image, (35, 70), 8, (0, 0, 255), -1)               # Red light
    cv2.circle(image, (35, 100), 8, (0, 255, 255), -1)            # Yellow light
    cv2.circle(image, (35, 130), 8, (0, 255, 0), -1)              # Green light

    return image

def main():
    # Set up the project paths using `pathlib.Path` for cross-platform compatibility
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    data_dir = project_root / "data" / "sample_images"
    
    # ========================================================================
    # Load YOLO Model
    # ========================================================================
    print("Loading pretrained YOLOv8 model...")
    model = load_pretrained_model()
    print(f"Model loaded: {model.model.__class__.__name__}")
    print(f"Classes available: {len(model.names)} (COCO dataset)")
    
    # Print some COCO classes
    coco_classes = list(model.names.values())
    print(f"\nCOCO classes: {', '.join(coco_classes[:10])}, ...")
    print("Note: Solar panels are NOT in this list!")
    
    # Look for real images in the data directory. If we find any, we'll use the
    # first one. Otherwise, we'll fall back to the synthetic test image.
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    
    if image_files:
        print(f"\nUsing {image_files[0].name}")
        image = load_image(image_files[0])
    else:
        print("\nNo images found, using synthetic test image")
        image = create_test_image_with_common_objects()
    
    # ========================================================================
    # Run Inference
    # ========================================================================
    print("\nRunning YOLO inference...")
    results = run_inference(model, image, conf_threshold=0.25)
    
    print()
    analyze_detection_results(results)
    
    # ========================================================================
    # Visualize Results
    # ========================================================================
    print("\nVisualizing results...")
    visualize_results(image, results, output_dir / "07_yolo_detections.png")
    print(f"Saved {output_dir / '07_yolo_detections.png'}")

    print("\nVisualizing NMS effect...")
    visualize_nms(model, image, output_dir / "08_nms_effect.png")
    print(f"Saved {output_dir / '08_nms_effect.png'}")

if __name__ == "__main__":
    main()