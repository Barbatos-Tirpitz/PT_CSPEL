import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Default values for various colors
colors = {
    "Red": {
        "lower": [0, 50, 50],
        "upper": [10, 255, 255],
        "highlight": (255, 0, 0)  # Add highlight color for Red
    },
    "Green": {
        "lower": [40, 40, 40],
        "upper": [80, 255, 255],
        "highlight": (0, 255, 0)  # Add highlight color for Green
    },
    "Blue": {
        "lower": [100, 50, 50],
        "upper": [130, 255, 255],
        "highlight": (0, 0, 255)  # Add highlight color for Blue
    },
    "Yellow": {
        "lower": [20, 100, 100],
        "upper": [30, 255, 255],
        "highlight": (255, 255, 0)  # Add highlight color for Yellow
    },
    "Orange": {
        "lower": [5, 100, 100],
        "upper": [15, 255, 255],
        "highlight": (255, 165, 0)  # Add highlight color for Orange
    },
    "Purple": {
        "lower": [125, 50, 50],
        "upper": [155, 255, 255],
        "highlight": (128, 0, 128)  # Add highlight color for Purple
    }
    # Add more colors as needed
}

def select_image():
    global cv_image, label_image, file_path
    
    # select an image
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
        label_image.pack(side=tk.RIGHT)  # Pack the label on the right side

def remove_highlights():
    global cv_image, file_path
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv_image = cv2.imread(file_path)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(cv_image)
    image_tk = ImageTk.PhotoImage(image_pil)
    label_image.configure(image=image_tk)
    label_image.image = image_tk

def process_image(selected_color, highlight_mode, color_tolerance, segmentation_threshold, lower_bound_values, upper_bound_values, kernel_size, transparency):
    global cv_image
    
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
    
    # Get lower and upper bounds for the selected color
    lower_bound = np.array(lower_bound_values)
    upper_bound = np.array(upper_bound_values)
    
    # Create a mask for the selected color with tolerance
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Adjust kernel size based on user input
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image with transparency
    highlight_color = colors[selected_color]["highlight"]
    overlay = cv_image.copy()
    cv2.fillPoly(overlay, contours, highlight_color)
    cv2.addWeighted(overlay, transparency, cv_image, 1 - transparency, 0, cv_image)
    
    # Resize the processed image if dimensions exceed 1280x720
    height, width, _ = cv_image.shape
    if height > 720 or width > 1280:
        cv_image = cv2.resize(cv_image, (1280, 720))
    
    # Convert image from RGB to BGR for displaying with OpenCV
    cv_image_display = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    # Display the processed image
    cv2.imshow('Processed Image', cv_image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clear_image():
    global cv_image
    cv_image = None
    label_image.config(image=None)
    label_image.pack_forget()  

def refresh():
    root.update()

# Create Tkinter window
root = tk.Tk()
root.title("Image Color Detection")

# Set window resolution
root.geometry("1520x720")

# Create frame for left side GUI elements
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT)

# Dropdown menu for selecting color
selected_color = tk.StringVar(left_frame)
selected_color.set("Red")  # "Set" Default selection
color_options = list(colors.keys())  # Get color options from dictionary
color_menu = tk.OptionMenu(left_frame, selected_color, *color_options)
color_menu.pack()

# Dropdown menu for selecting highlight mode
highlight_mode = tk.StringVar(left_frame)
highlight_mode.set("Full")  # "Set" Default selection
highlight_options = ["Full", "Boundary"]
highlight_menu = tk.OptionMenu(left_frame, highlight_mode, *highlight_options)
highlight_menu.pack()

# Color tolerance slider
color_tolerance_label = tk.Label(left_frame, text="Color Tolerance:")
color_tolerance_label.pack()
color_tolerance_scale = tk.Scale(left_frame, from_=0, to=100, orient=tk.HORIZONTAL)
color_tolerance_scale.set(30)  # "set" Adjust default value for better color detection
color_tolerance_scale.pack()

# Segmentation threshold slider
segmentation_threshold_label = tk.Label(left_frame, text="Segmentation Threshold:")
segmentation_threshold_label.pack()
segmentation_threshold_scale = tk.Scale(left_frame, from_=0, to=255, orient=tk.HORIZONTAL)
segmentation_threshold_scale.set(100)  # "set" Default value
segmentation_threshold_scale.pack()

# Kernel size slider
kernel_size_label = tk.Label(left_frame, text="Kernel Size:")
kernel_size_label.pack()
kernel_size_scale = tk.Scale(left_frame, from_=1, to=15, orient=tk.HORIZONTAL)
kernel_size_scale.set(3)  # "set" Default kernel size
kernel_size_scale.pack()

# Transparency slider
transparency_label = tk.Label(left_frame, text="Transparency:")
transparency_label.pack()
transparency_scale = tk.Scale(left_frame, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL)
transparency_scale.set(0.5)  # Default transparency
transparency_scale.pack()

# Advanced options for adjusting lower and upper bounds
advanced_options_frame = tk.Frame(left_frame)
advanced_options_frame.pack()

# Function to update sliders based on color selection
def update_sliders(selected_color):
    lower_bound_values = colors[selected_color]["lower"]
    upper_bound_values = colors[selected_color]["upper"]
    for i, slider in enumerate(lower_bound_sliders):
        slider.set(lower_bound_values[i])
    for i, slider in enumerate(upper_bound_sliders):
        slider.set(upper_bound_values[i])

# Label for lower bound sliders
lower_bound_label = tk.Label(advanced_options_frame, text="Lower Bound:")
lower_bound_label.grid(row=0, column=0, padx=5, pady=5)

# Sliders for lower bounds
lower_bound_sliders = []
for i in range(3):
    lower_bound_slider = tk.Scale(advanced_options_frame, from_=0, to=255, orient=tk.HORIZONTAL)
    lower_bound_slider.grid(row=0, column=i+1, padx=5, pady=5)
    lower_bound_sliders.append(lower_bound_slider)

# Label for upper bound sliders
upper_bound_label = tk.Label(advanced_options_frame, text="Upper Bound:")
upper_bound_label.grid(row=1, column=0, padx=5, pady=5)

# Sliders for upper bounds
upper_bound_sliders = []
for i in range(3):
    upper_bound_slider = tk.Scale(advanced_options_frame, from_=0, to=255, orient=tk.HORIZONTAL)
    upper_bound_slider.grid(row=1, column=i+1, padx=5, pady=5)
    upper_bound_sliders.append(upper_bound_slider)

# Update sliders when color selection changes
selected_color.trace_add("write", lambda *args: update_sliders(selected_color.get()))

# Presets menu
preset_menu = tk.OptionMenu(left_frame, tk.StringVar(left_frame, "Presets"), "Presets", *color_options, command=update_sliders)
preset_menu.pack()

# Button to select an image
select_button = tk.Button(left_frame, text="Select Image", command=select_image)
select_button.pack()

# Button to remove highlights
remove_button = tk.Button(left_frame, text="Remove Highlights", command=remove_highlights)
remove_button.pack()

# Button to process selected image
process_button = tk.Button(left_frame, text="Process Image", command=lambda: process_image(selected_color.get(), highlight_mode.get(), color_tolerance_scale.get(), segmentation_threshold_scale.get(), [slider.get() for slider in lower_bound_sliders], [slider.get() for slider in upper_bound_sliders], kernel_size_scale.get(), transparency_scale.get()))
process_button.pack()

# Button to clear processed image
clear_button = tk.Button(left_frame, text="Clear", command=clear_image)
clear_button.pack()

# Button to refresh
refresh_button = tk.Button(left_frame, text="Refresh", command=refresh)
refresh_button.pack()

# Variable to store the loaded image
cv_image = None
file_path = None

# Label to display selected image
label_image = tk.Label(root)
label_image.pack(side=tk.RIGHT)

root.mainloop()
