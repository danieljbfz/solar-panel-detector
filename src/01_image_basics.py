"""
Image Basics

This script covers fundamental concepts in digital image processing. We'll look
at what images are at the data level, how color information is encoded, and why
edge detection matters for object recognition.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def load_image(image_path):
    """
    Loads an image from disk and converts it to RGB color space.
    
    OpenCV loads images in BGR format by default (a historical quirk from early
    camera sensor designs), so we need to convert the loaded BGR array to RGB,
    which is the standard format used by most other libraries (matplotlib,
    PIL, scikit-image) and matches how we typically think about color channels.
    
    :param image_path: Path to the image file on disk.
    :return: The image as a numpy array in RGB format.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR to RGB using OpenCV's color conversion function
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def analyze_image_structure(image):
    """
    Prints basic structural information about the image array.
    
    Images in NumPy are represented as 3D arrays with shape (height, width,
    channels). For an RGB image, we have 3 channels. The dtype tells us how
    pixel values are stored (typically `uint8` for 8-bit color images, meaning
    each channel value ranges from 0 to 255). Memory usage is calculated from
    the total number of bytes in the array, which matters when processing large
    batches of high-resolution images.
    
    :param image: The image array to analyze.
    """
    print(f"Shape: {image.shape}")
    print(f"  → Height (rows): {image.shape[0]} pixels")
    print(f"  → Width (cols):  {image.shape[1]} pixels")
    print(f"  → Channels:      {image.shape[2]} (RGB)")
    
    print(f"\nData type: {image.dtype}")
    print(f"  → uint8 means: unsigned integer, 0-255 range")
    print(f"  → This is standard for 8-bit color images")
    
    print(f"\nValue range: [{image.min()}, {image.max()}]")
    
    memory_bytes = image.nbytes
    memory_mb = memory_bytes / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    print(f"  → Calculation: {image.shape[0]} × {image.shape[1]} × {image.shape[2]} = {memory_bytes:,} bytes")

    # A single 20MP drone image = ~60 MB
    # A mission with 1000 images = 60 GB of raw data to process!

def visualize_color_channels(image, output_path):
    """
    Splits the image into its R, G, B channels and visualizes them separately.
    
    Each pixel in an RGB image stores three values, one for each color channel.
    By looking at the channels separately, we can see how color information is
    distributed. For example, objects that appear blue in the original image
    will have high values in the blue channel but lower values in the red and
    green channels.
    
    :param image: The RGB image array.
    :param output_path: Path where the figure will be saved.
    """
    # Array slicing syntax: image[:, :, 0] means "all rows, all columns, first channel"
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    
    # We use colormaps like 'Reds', 'Greens', 'Blues' to visualize each channel
    axes[0, 1].imshow(red_channel, cmap='Reds')
    axes[0, 1].set_title('Red Channel')
    
    axes[1, 0].imshow(green_channel, cmap='Greens')
    axes[1, 0].set_title('Green Channel')
    
    axes[1, 1].imshow(blue_channel, cmap='Blues')
    axes[1, 1].set_title('Blue Channel')

    for ax in axes.flatten(): ax.axis('off')    

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def convert_to_grayscale(image):
    """
    Converts an RGB image to grayscale using a weighted sum.
    
    Instead of taking a simple average of the three channels, we use the
    formula 0.299*R + 0.587*G + 0.114*B. These weights are based on human
    perception. Our eyes are more sensitive to green light than red or blue, so
    the green channel gets more weight. This formula preserves perceived
    brightness better than an unweighted average.
    
    :param image: The RGB image array.
    :return: The grayscale image as a 2D array.
    """
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # We convert the result back to uint8 (8-bit integers) to match the original image format
    return gray.astype(np.uint8)


def compute_gradients(gray_image):
    """
    Calculates image gradients to detect edges.
    
    Edges occur where pixel intensity changes rapidly. The Sobel operator
    computes the derivative of the image in both the x direction (horizontal
    edges) and the y direction (vertical edges) using a 3x3 convolution kernel.
    We then combine these two gradients using the Euclidean distance formula to
    get the overall edge magnitude at each pixel.
    
    This is conceptually similar to what happens in the first layers of a
    convolutional neural network. CNNs learn their own edge detectors during
    training, but they're fundamentally doing the same kind of operation.
    
    :param gray_image: The grayscale image array.
    :return: The edge magnitude array.
    """
    # `cv2.Sobel` parameters: (input, output depth, derivative order in x, derivative order in y, kernel size)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine the x and y gradients using Euclidean distance
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Clip values to the valid range and convert back to uint8
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude


def visualize_edge_detection(image, output_path):
    """
    Demonstrates edge detection on the image and saves the result.
    
    :param image: The RGB image array.
    :param output_path: Path where the figure will be saved.
    """
    gray = convert_to_grayscale(image)
    edges = compute_gradients(gray)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale')
    
    axes[2].imshow(edges, cmap='hot')
    axes[2].set_title('Edges (Sobel)')
    
    for ax in axes: ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def demonstrate_color_thresholding(image, output_path):
    """
    Attempts to segment solar panels using color thresholding alone.
    
    This is a simple segmentation technique where we convert the image to HSV
    color space (which separates hue, saturation, and value) and then threshold
    the hue channel to find pixels in a specific color range. In this case,
    we're looking for blue pixels, which is where solar panels typically fall.
    
    The reason we use HSV instead of RGB is that HSV separates the color
    information (hue) from the brightness (value), making it more robust to
    lighting changes. However, even with HSV, this method is still fragile. It
    fails when there are shadows, reflections, or panels of different colors.
    
    :param image: The RGB image array.
    :param output_path: Path where the figure will be saved.
    :return: The percentage of the image that was classified as blue.
    """
    # Convert from RGB to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define the range for blue hues. In OpenCV's HSV representation, hue is
    # represented in the range [0, 180] (not [0, 360] as in standard HSV),
    # saturation is [0, 255], and value is [0, 255].
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create a binary mask where pixels within the blue range are set to 255
    # (white) and all other pixels are set to 0 (black)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original')
    
    # Force the plotter to use the full 0-180 range
    axes[1].imshow(hsv[:, :, 0], cmap='hsv', vmin=0, vmax=180)
    axes[1].set_title('Hue Channel')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Blue Mask')
    
    for ax in axes: ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate what percentage of the image was classified as blue
    coverage = (mask > 0).sum() / mask.size * 100
    return coverage


def create_synthetic_test_image():
    """
    Creates a simple synthetic test image for initial experimentation.
    
    This image simulates an aerial view with a green background (representing
    grass or ground) and two blue rectangles (representing solar panels). We
    also add some lighter blue spots to simulate glare or reflections.
    
    Once we have real data, the script will automatically use that instead.
    
    :return: A synthetic RGB image as a numpy array.
    """
    height, width = 400, 600
    
    # Initialize a blank image (all zeros) with 3 channels
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill the entire image with a green color [R, G, B]
    image[:, :] = [100, 150, 80]
    
    # Draw two blue rectangles to represent solar panels
    image[100:200, 150:300] = [40, 60, 120]
    image[150:250, 350:480] = [35, 55, 115]
    
    # Add some lighter blue spots to simulate glare
    image[120:140, 180:220] = [80, 100, 160]
    image[170:190, 380:420] = [75, 95, 155]
    
    return image

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
        print("No images found, using synthetic test image")
        image = create_synthetic_test_image()
    
    print()
    analyze_image_structure(image)
    
    # Generate all three visualizations and save them to the outputs directory
    print("\nGenerating visualizations...")
    visualize_color_channels(image, output_dir / "01_color_channels.png")
    print(f"Saved {output_dir / '01_color_channels.png'}")
    
    visualize_edge_detection(image, output_dir / "02_edges.png")
    print(f"Saved {output_dir / '02_edges.png'}")
    
    coverage = demonstrate_color_thresholding(image, output_dir / "03_color_threshold.png")
    print(f"Saved {output_dir / '03_color_threshold.png'}")
    print(f"Color thresholding detected {coverage:.1f}% of the image as blue")


if __name__ == "__main__":
    main()