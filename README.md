# Cat-eye-detection

# Overview
This Python script shows how to recognise a cat's eye using Haar Cascade classifiers and convert pixel positions to (x, y) coordinates. The script additionally marks the detected eye and custom coordinates on the image.

# Requirements
- Python 3.x
- OpenCV library
- Google Colab environment (optional, for `cv2_imshow`)

# Installation
1. Install the OpenCV library if not already installed:
    ```bash
    pip install opencv-python-headless
    ```
2. Download the script (`cat_eye_detection.py`) and save it to your local directory.

# Usage
1. Update the `image_path` in the script to the path of your image file.
2. Run the script.

# Example Usage
```python
if __name__ == "__main__":
    image_path = "/path/to/your/image.jpg"
    eye_coordinates = get_eye_coordinates(image_path)
    custom_coordinates = (150, 160)  # Custom coordinates

    if eye_coordinates is not None:
        x, y = eye_coordinates
        print(f"Eye pixel location: (px: {x}, py: {y})")
        print(f"Cat's left eye approximate center coordinates: (x: {x}, y: {y})")

        # Convert the eye coordinates to pixel values
        px, py = pixel_to_coordinates(x, y)
        print(f"Eye pixel location: (px: {px}, py: {py})")

        # Convert the custom coordinates to pixel values
        px_custom, py_custom = pixel_to_coordinates(custom_coordinates[0], custom_coordinates[1])
        print(f"Custom pixel location: (px: {px_custom}, py: {py_custom})")
        print(f"Custom coordinates: (x: {custom_coordinates[0]}, y: {custom_coordinates[1]})")

        display_cat_face(image_path, eye_coordinates, custom_coordinates)
    else:
        print("Failed to detect eye.")
```

# Assumptions
1.The input image is a cat's face where the eyes are clearly visible.

2.The image format is supported by OpenCV (e.g., JPEG, PNG).

# Note
1.The script uses the Haar Cascade classifier provided by OpenCV, which is pre-trained for general eye detection

2.The above respositry contains two codes i.e One for Normal Python Script(.py script), which can used for IDE's like VS Code etc. and the other for Juypter Notebook(.ipynb script), which can be used for IDE's like Google Colab etc. 

# Refernce
OpenCV Documentation (https://docs.opencv.org)
