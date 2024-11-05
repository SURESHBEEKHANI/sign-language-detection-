# Import necessary libraries
import os               # Provides a way of using operating system dependent functionality like reading or writing to the filesystem
import cv2              # OpenCV library for computer vision tasks
import time             # Provides various time-related functions
import uuid             # Provides functions to generate unique identifiers

# Define the path where collected images will be stored
IMAGE_PATH = "CollectedImages"

# List of labels for sign language gestures to be captured
labels = ['Hello', 'Yes', 'No', 'Thanks', 'IloveYou', 'Please']

# Define the number of images to capture for each label
number_of_images = 5

# Loop through each label to create directories and capture images
for label in labels:
    # Create a path for each label's images
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path)  # Create the directory for the label if it doesn't exist

    # Open the camera using OpenCV
    cap = cv2.VideoCapture(0)  # 0 refers to the default camera
    print(f"Collecting images for {label}")  # Inform user about the current label being collected
    time.sleep(3)  # Wait for 3 seconds before starting to capture images

    # Capture the specified number of images
    for imgnum in range(number_of_images):
        ret, frame = cap.read()  # Read a frame from the camera
        # Construct the full image path using the label and a unique filename
        imagename = os.path.join(IMAGE_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename, frame)  # Save the captured frame as an image file
        cv2.imshow('frame', frame)  # Display the captured frame in a window
        time.sleep(2)  # Wait for 2 seconds before capturing the next image

        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed
    
    cap.release()  # Release the camera resource after capturing images
