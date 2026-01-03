"""
Classical Detection Methods

This script explores traditional computer vision techniques for object detection
that predate deep learning. We'll look at keypoint detection algorithms (SIFT and 
ORB), template matching, and contour-based segmentation. These methods help us 
understand what features neural networks might be learning automatically.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


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

def detect_sift_keypoints(image):
    """
    Detects keypoints and computes descriptors using the SIFT algorithm.
    
    SIFT (Scale-Invariant Feature Transform) identifies distinctive points in
    an image that are likely to be recognizable even when the image is scaled,
    rotated, or viewed from a different angle. Each keypoint comes with a
    descriptor (a 128-dimensional vector) that encodes the local appearance
    around that point.

    The algorithm works in a series of steps:
    1. Build a scale-space pyramid (the image at multiple resolutions)
    2. Find local extrema in the difference-of-Gaussian (DoG) images
    3. Refine keypoint locations and filter out weak ones
    4. Assign orientations based on local gradient directions
    5. Compute descriptors (128-dimensional vectors) around each keypoint
    
    These keypoints are useful for matching objects across different images,
    even when the viewing angle, lighting, or scale changes. However, SIFT is
    patented (though the patent has expired in some regions) and can be slow to
    compute, which is why faster alternatives like ORB were developed later.
    
    :param image: The RGB image array.
    :return: A tuple of (keypoints, descriptors, image_with_keypoints).
    """
    # Convert the image to grayscale since SIFT operates on intensity values
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create the SIFT detector object
    sift = cv2.SIFT_create()
    
    # The `detectAndCompute` method finds keypoints and computes their
    # descriptors in a single pass. `keypoints` is a list of `cv2.KeyPoint` 
    # objects, each containing position, size, angle, and response (strength). 
    # `descriptors` is a numpy array of shape (num_keypoints, 128) where each 
    # row is a 128-dimensional feature vector.
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def detect_orb_keypoints(image):
    """
    Detects keypoints and computes descriptors using the ORB algorithm.
    
    ORB (Oriented FAST and Rotated BRIEF) is a fast, free alternative to SIFT.
    It was designed by OpenCV researchers specifically to avoid patent issues
    and to be computationally efficient enough for real-time applications. 
    
    ORB combines two algorithms:
    1. FAST (Features from Accelerated Segment Test) for keypoint detection
    2. BRIEF (Binary Robust Independent Elementary Features) for descriptors

    The main difference from SIFT is that ORB produces binary descriptors (each
    descriptor is a 256-bit binary string) instead of floating-point vectors.
    This makes it much faster because we can use the Hamming distance (XOR and 
    popcount) instead of the Euclidean distance (L2 norm) when comparing
    different descriptors.
    
    :param image: The RGB image array.
    :return: A tuple of (keypoints, descriptors, image_with_keypoints).
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create an ORB detector with the maximum number of keypoints set to 1000.
    # (default is 500). By increasing the number of keypoints, we can expect
    # better coverage of the image, though it will take longer to compute.
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Similar to SIFT, we use `detectAndCompute` to get both keypoints and
    # descriptors. ORB descriptors are binary (each element is 0 or 1) and
    # much shorter than SIFT descriptors.
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def visualize_keypoints(image, sift_kp, orb_kp, output_path):
    """
    Draws keypoints on the image for SIFT and ORB and saves the result.

    Keypoints are visualized as circles. The size of each circle represents the
    scale (how large the feature is), and the orientation line shows the
    dominant gradient direction at that point.
    
    :param image: The RGB image array.
    :param sift_kp: List of `cv2.KeyPoint` objects.
    :param orb_kp: List of `cv2.KeyPoint` objects.
    :param output_path: Path where the figure will be saved.
    """
    # The `drawKeypoints` function draws keypoints on the image. The flag
    # `cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` tells OpenCV to draw both
    # the scale (circle size) and orientation (line direction) of each keypoint.
    sift_img = cv2.drawKeypoints(
        image, 
        sift_kp, 
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # The `drawKeypoints` function is used again for ORB. Here, we pass a specific 
    # color (green) and use `flags=0`, which draws only the location of the 
    # keypoints as small dots on the image (no scale or orientation).
    orb_img = cv2.drawKeypoints(
        image,
        orb_kp,
        None,
        color=(0, 255, 0),
        flags=0
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')

    axes[1].imshow(sift_img)
    axes[1].set_title(f'SIFT ({len(sift_kp)} pts)')

    axes[2].imshow(orb_img)
    axes[2].set_title(f'ORB ({len(orb_kp)} pts)')

    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def detect_contours(image, min_area):
    """
    Detects contours in the image using a multi-stage computer vision pipeline.
        
    This pipeline isolates structural features by identifying intensity gradients 
    and grouping them into continuous vector paths (contours).  
    
    The pipeline works as follows:
    1. Grayscale: Removes color from the image so we can focus on intensity gradients.
    2. Blur: Convolves the image with a kernel to suppress high-frequency noise.
    3. Edges: Uses the Canny algorithm to trace the sharpest outlines in the image.
    4. Contours: Connects those edge pixels into closed shapes (vector paths).
    5. Filtering: Removes any shapes that are smaller than the `min_area` threshold.

    This method can work well for objects with clear boundaries against a
    contrasting background, but it struggles with complex scenes, shadows, and
    overlapping objects.
    
    :param image: The RGB image array.
    :param min_area: The minimum contour area in pixels (filters out noise).
    :return: A tuple (stages, filtered_contours) where:
             - stages: A list of dictionaries containing the title, image 
               array, and colormap for each processing step.
             - filtered_contours: A list of numpy arrays, each representing 
               the (x, y) coordinates of a detected boundary.
    """
    # Define stages as a list of dictionaries to store the images and titles
    stages = []

    # Stage 0: Original image
    stages.append({"title": "Original", "image": image, "cmap": None})

    # Stage 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # stages.append({"title": "Grayscale", "image": gray, "cmap": "gray"})

    # Stage 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    stages.append({"title": "Blurred", "image": blurred, "cmap": "gray"})

    # Stage 3: Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    stages.append({"title": "Canny Edges", "image": edges, "cmap": "gray"})

    # Stage 4: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours that are likely noise
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Draw the contours on the original image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, filtered, -1, (0, 255, 0), 2)
    stages.append({"title": f"Contours ({len(filtered)})", "image": image_with_contours, "cmap": None})

    return stages, filtered

def visualize_contours(stages, output_path):
    """
    Draws the contours on the image and saves the result.
    
    :param stages: A list of dictionaries containing the title, image array, 
                   and colormap for each processing step in the pipeline.
    :param output_path: Path where the figure will be saved.
    """
    n = len(stages)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(15, 10))
    
    for ax, stage in zip(axes.flatten(), stages):
        ax.imshow(stage["image"], cmap=stage.get("cmap"))
        ax.set_title(stage["title"])
        ax.axis('off')
        
    # Hide any unused subplots if n is odd
    for i in range(n, len(axes.flatten())):
        axes.flatten()[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def template_matching(image, template):
    """
    Attempts to find the template within the larger image.
    
    Template matching is the simplest form of object detection. We slide the
    template across the image and compute a similarity score at each position.
    The position with the highest score is considered the match.
    
    The problem with template matching is that it only works when:
    1. The template and object are at the same scale
    2. The template and object have the same orientation
    3. The lighting conditions are similar
    4. There's no occlusion or distortion
    
    This is why template matching is rarely used in production systems. It's
    too brittle to be useful in real-world applications.
    
    :param image: The larger image to search in.
    :param template: The smaller template image to find.
    :return: A tuple of (match_location, confidence) where match_location is
             (x, y) and confidence is the correlation score.
    """
    # Convert both images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    
    # `cv2.matchTemplate` computes the correlation between the template and
    # every possible position in the image. We use `cv2.TM_CCOEFF_NORMED` to
    # compute the correlation coefficient, normalized to the range [-1, 1],
    # where 1 is a perfect match
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    
    # Find the location of the best match (highest correlation coefficient)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # `max_loc` gives us the top-left corner of the match
    # `max_val` is the correlation coefficient
    return max_loc, max_val

def visualize_template_match(image, template, match_location, confidence, output_path):
    """
    Draws a rectangle around the matched template and saves the result.
    
    :param image: The RGB image array.
    :param template: The template that was matched.
    :param match_location: The (x, y) coordinates of the top-left corner.
    :param confidence: The correlation score.
    :param output_path: Path where the figure will be saved.
    """
    # Get the template dimensions
    h, w = template.shape[:2]
    top_left = match_location
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Draw a rectangle on a copy of the image
    img_with_box = image.copy()
    cv2.rectangle(img_with_box, top_left, bottom_right, (0, 255, 0), 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original')

    axes[1].imshow(template)
    axes[1].set_title('Template')
    
    axes[2].imshow(img_with_box)
    axes[2].set_title(f'Match (confidence: {confidence:.2f})')

    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_synthetic_test_image():
    """
    Creates a synthetic test image with multiple objects.

    :return: A synthetic RGB image as a numpy array.
    """
    height, width = 400, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill the entire image with a green color
    image[:, :] = [100, 150, 80]
    
    # Add some solar panels (different sizes, positions, and colors)
    image[80:160, 120:280] = [40, 60, 120]
    image[140:240, 320:460] = [35, 55, 115]
    image[250:320, 100:220] = [38, 58, 118]
    
    # Add some texture/noise to make it more realistic
    noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add glare spots
    image[100:120, 160:200] = [80, 100, 160]
    image[170:190, 360:400] = [75, 95, 155]
    
    return image

def create_template_from_image(image, x, y, width, height):
    """
    Extracts a rectangular region from the image to use as a template.
    
    :param image: The RGB image array.
    :param x: X coordinate of the top-left corner.
    :param y: Y coordinate of the top-left corner.
    :param width: Width of the template.
    :param height: Height of the template.
    :return: The extracted template as a numpy array.
    """
    return image[y:y+height, x:x+width].copy()

def main():
    # Set up the project paths using `pathlib.Path` for cross-platform compatibility
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    data_dir = project_root / "data" / "sample_images"
    
    # Look for real images in the data directory. If we find any, we'll use the
    # first one. Otherwise, we'll fall back to the synthetic test image.
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    
    if image_files:
        print(f"Using {image_files[0].name}")
        image = load_image(image_files[0])
    else:
        print("No images found, using synthetic test scene")
        image = create_synthetic_test_image()
    
    print()

    # ========================================================================
    # Keypoint Detection (SIFT and ORB)
    # ========================================================================
    print("\nDetecting SIFT keypoints...")
    sift_keypoints, sift_descriptors = detect_sift_keypoints(image)
    print(f"Found {len(sift_keypoints)} SIFT keypoints")
    
    print("\nDetecting ORB keypoints...")
    orb_keypoints, orb_descriptors = detect_orb_keypoints(image)
    print(f"Found {len(orb_keypoints)} ORB keypoints")
    
    visualize_keypoints(
        image,
        sift_keypoints,
        orb_keypoints,
        output_dir / "04_keypoints.png"
    )
    print(f"Saved {output_dir / '04_keypoints.png'}")

    # ========================================================================
    # Contour Detection
    # ========================================================================
    print("\nDetecting contours...")
    stages, contours = detect_contours(image, min_area=500)
    print(f"Found {len(contours)} contours (area > 500 pixels)")
    
    visualize_contours(stages, output_dir / "05_contours.png")
    print(f"Saved {output_dir / '05_contours.png'}")

    # ========================================================================
    # Template Matching
    # ========================================================================
    print("\nPerforming template matching...")
    # For template matching, we'll extract a small region from the image and
    # try to find it. In a real scenario, we would have a separate template image.
    template = create_template_from_image(image, x=130, y=90, width=90, height=60)
    location, confidence = template_matching(image, template)
    print(f"Best match at {location} with confidence {confidence:.2f}")
    
    visualize_template_match(
        image, 
        template, 
        location,
        confidence,
        output_dir / "06_template_match.png"
    )
    print(f"Saved {output_dir / '06_template_match.png'}")

if __name__ == "__main__":
    main()