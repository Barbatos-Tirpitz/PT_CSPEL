import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def select_image():
    global cv_image, label_image, file_path
    
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load selected image
        cv_image = cv2.imread(file_path)
        
        if cv_image is None:
            print("Error: Failed to load image")
            return
        
        # Convert image from BGR to RGB for displaying in Tkinter
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
        label_image.image = image_tk  # Keep a reference to prevent garbage collection
        label_image.pack()

def remove_highlights():
    global cv_image, file_path
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv_image = cv2.imread(file_path)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(cv_image)
    image_tk = ImageTk.PhotoImage(image_pil)
    label_image.configure(image=image_tk)
    label_image.image = image_tk

def process_image(selected_color, highlight_mode, color_tolerance, segmentation_threshold):
    global cv_image
    
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
    
    # Define lower and upper bounds for the selected color
    if selected_color == "Red":
        lower_bound = np.array([0 - color_tolerance, 50, 50])  # Lower bound for red color
        upper_bound = np.array([10 + color_tolerance, 255, 255])  # Upper bound for red color
        highlight_color = (255, 0, 0)  # Red color for highlighting
    elif selected_color == "Green":
        lower_bound = np.array([40 - color_tolerance, 40, 40])  # Lower bound for green color
        upper_bound = np.array([80 + color_tolerance, 255, 255])  # Upper bound for green color
        highlight_color = (0, 255, 0)  # Green color for highlighting
    elif selected_color == "Blue":
        lower_bound = np.array([100 - color_tolerance, 50, 50])  # Lower bound for blue color
        upper_bound = np.array([130 + color_tolerance, 255, 255])  # Upper bound for blue color
        highlight_color = (0, 0, 255)  # Blue color for highlighting
    else:
        print("Invalid color selection")
        return
    
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    
    # Create a mask for the selected color with tolerance
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    print("Mask:", mask)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    print("Mask after morphology:", mask)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("Contours:", contours)
    
    # Draw contours on the original image
    if highlight_mode == "Full":
        cv2.fillPoly(cv_image, contours, highlight_color)
    elif highlight_mode == "Boundary":
        cv2.drawContours(cv_image, contours, -1, highlight_color, 2)
    
    # Convert image from RGB to BGR for displaying with OpenCV
    cv_image_display = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    # Display results
    cv2.imshow('Processed Image', cv_image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clear_image():
    global cv_image
    cv_image = None
    label_image.config(image=None)
    label_image.pack_forget()  # Remove the label widget from the GUI layout

# Create Tkinter window
root = tk.Tk()
root.title("Image Color Detection")

# Set window resolution to 1280x720 pixels
root.geometry("1280x720")

# Dropdown menu for selecting color
selected_color = tk.StringVar(root)
selected_color.set("Red")  # Default selection
color_options = ["Red", "Green", "Blue"]
color_menu = tk.OptionMenu(root, selected_color, *color_options)
color_menu.pack()

# Dropdown menu for selecting highlight mode
highlight_mode = tk.StringVar(root)
highlight_mode.set("Full")  # Default selection
highlight_options = ["Full", "Boundary"]
highlight_menu = tk.OptionMenu(root, highlight_mode, *highlight_options)
highlight_menu.pack()

# Color tolerance slider
color_tolerance_label = tk.Label(root, text="Color Tolerance:")
color_tolerance_label.pack()
color_tolerance_scale = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL)
color_tolerance_scale.set(50)  # Default value
color_tolerance_scale.pack()

# Segmentation threshold slider
segmentation_threshold_label = tk.Label(root, text="Segmentation Threshold:")
segmentation_threshold_label.pack()
segmentation_threshold_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL)
segmentation_threshold_scale.set(100)  # Default value
segmentation_threshold_scale.pack()

# Button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

# Button to remove highlights
remove_button = tk.Button(root, text="Remove Highlights", command=remove_highlights)
remove_button.pack()

# Label to display selected image
label_image = tk.Label(root)
label_image.pack()

# Button to process selected image
process_button = tk.Button(root, text="Process Image", command=lambda: process_image(selected_color.get(), highlight_mode.get(), color_tolerance_scale.get(), segmentation_threshold_scale.get()))
process_button.pack()

# Button to clear processed image
clear_button = tk.Button(root, text="Clear", command=clear_image)
clear_button.pack()

# Variable to store the loaded image
cv_image = None
file_path = None

root.mainloop()
