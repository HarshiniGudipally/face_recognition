import os
from PIL import Image

# Define the path to the folder containing images
input_folder = 'C:\\Users\\harsh\\Work\\face_recognition\\data\\reception_jpgs'
output_folder = 'C:\\Users\\harsh\\Work\\face_recognition\\data\\reception_resized'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)
print(os.listdir(input_folder))
# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif','.JPG')):
        # Open an image file
        with Image.open(os.path.join(input_folder, filename)) as img:
            # Calculate new dimensions (half the original size)
            new_width = img.width // 2
            new_height = img.height // 2
            # Resize the image
            resized_img = img.resize((new_width, new_height))
            # Save the resized image to the output folder
            resized_img.save(os.path.join(output_folder, filename))

print("All images have been resized and saved.")
