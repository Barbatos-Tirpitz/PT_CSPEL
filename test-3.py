import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.python.keras.layers import *


# Global variables for image and file path
cv_image = None
label_image = None
file_path = None
# Path to the directory containing the SavedModel
saved_model_dir = r'C:\Users\Razgr\Downloads\mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8\mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8\saved_model'

# Load the SavedModel
loaded_model = tf.saved_model.load(saved_model_dir)

# Get the inference function from the loaded model
inference_func = loaded_model.signatures["serving_default"]

def select_image():
    global cv_image, label_image, file_path
    
    # Select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load image
        cv_image = cv2.imread(file_path)
        
        if cv_image is None:
            print("Error: Failed to load image")
            return
        
        # Convert image to display
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Convert image to PIL format
        image_pil = Image.fromarray(cv_image)
        # Convert PIL image to Tkinter format
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Remove previous label widget if exists
        if label_image:
            label_image.destroy()
        
        # Create a new label widget to display the image
        label_image = tk.Label(root, image=image_tk)
        label_image.image = image_tk  
        label_image.grid(row=0, column=1, padx=10, pady=10)  # Display the image on the right side

def segment_image(image):
    # Preprocess image (resize, normalize, etc.)
    image = preprocess_image(image)
    
    # Perform inference with the model
    predictions = model.predict(image)
    
    # Extract segmentation masks from predictions
    masks = predictions['masks']
    
    return masks

def preprocess_image(image):
    # Resize image to match model input size
    image = cv2.resize(image, (1024, 1024))  # Adjust the size as per your model's input size
    
    # Normalize pixel values
    image = image / 255.0
    
    # Convert image to TensorFlow tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    
    return image

def apply_segmentation():
    global cv_image
    
    if cv_image is not None:
        # Preprocess image
        preprocessed_image = preprocess_image(cv_image)
        
        # Perform segmentation using DirectML backend
        with tf.device("/gpu:0"):
            masks = segment_image(preprocessed_image)
        
        # You can continue your processing and visualization here
        # Note: DirectML backend is used automatically for GPU acceleration

    if cv_image is not None:
        # Get parameter values from sliders
        transparency = transparency_scale.get()
        color_tolerance = color_tolerance_scale.get()
        threshold = threshold_scale.get()
        kernel_size = kernel_size_scale.get()
        lower_bound = lower_bound_scale.get()
        upper_bound = upper_bound_scale.get()

        # Apply image preprocessing
        blurred = cv2.blur(cv_image, (kernel_size, kernel_size))
        segmented = cv2.inRange(blurred, (lower_bound, lower_bound, lower_bound), 
                                (upper_bound, upper_bound, upper_bound))
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        result = cv_image.copy()
        colored_result = np.zeros_like(cv_image)

        # Assign a random color to each contour
        for contour in contours:
            color = np.random.randint(0, 255, size=3).tolist()  # Random color
            cv2.drawContours(colored_result, [contour], -1, color, -1)  # Fill contour with color

        # Combine original image and colored image with transparency
        overlay = cv2.addWeighted(cv_image, transparency / 100, colored_result, 1 - transparency / 100, 0)

        # Display the result
        cv2.imshow('Segmentation', overlay)

def main():
    global root
    
    # Create the main tkinter window
    root = tk.Tk()
    root.title("Image Segmentation")

    # Check if DirectML GPU support is available
    if 'DirectML' in [d.device_type for d in tf.config.list_physical_devices('GPU')]:
        print("DirectML support is available.")
    else:
        print("DirectML support is not available.")
        # Handle this case accordingly, perhaps fall back to CPU or display a warning
    
    # Frame for image selection
    frame_left = tk.Frame(root)
    frame_left.grid(row=0, column=0, padx=10, pady=10)
    
    # Button to select an image
    select_button = tk.Button(frame_left, text="Select Image", command=select_image)
    select_button.grid(row=0, column=0, padx=10, pady=10)
    
    # Button to apply segmentation
    apply_button = tk.Button(frame_left, text="Apply Segmentation", command=apply_segmentation)
    apply_button.grid(row=1, column=0, padx=10, pady=10)
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
