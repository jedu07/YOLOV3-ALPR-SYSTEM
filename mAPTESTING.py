import os
import subprocess
import re

# Directory containing the images
image_dir = './data/TEST/'

# Create results folder if it doesn't exist
if not os.path.exists('results-bbox'):
    os.makedirs('results-bbox')

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Sort image files
image_files.sort()

def process_image(image_file, detect_script, weights, size, model, image_dir):
    image_path = os.path.join(image_dir, image_file)
    result_folder = './results-bbox'

    cmd = f'python {detect_script} --weights {weights} --size {size} --model {model} --images {image_path} --info --dont_show'

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    confidence = None
    bbox_coords = None

    # Parse the output to extract bounding box coordinates and confidence scores
    for line in result.stdout.split('\n'):
        if 'Object found:' in line and 'Confidence:' in line and 'BBox Coords' in line:
            # Extract confidence and bounding box coordinates from the line using regular expressions
            match = re.search(r'Confidence: (\d+\.\d+)', line)
            if match:
                confidence = float(match.group(1))
            match = re.search(r'BBox Coords \(xmin, ymin, xmax, ymax\): (\d+\.\d+), (\d+\.\d+), (\d+\.\d+), (\d+\.\d+)', line)
            if match:
                bbox_coords = tuple(map(float, match.groups()))

    return confidence, bbox_coords, result_folder, image_file


detect_script = 'detect.py'
weights = './checkpoints/yolov3-custom-416'
size = '416'
model = 'yolov3'

total_images = len(image_files)

# Loop through each image file and process it
for image_file in image_files:
    confidence, bbox_coords, result_folder, image_name = process_image(image_file, detect_script, weights, size, model, image_dir)

    # Save confidence and bounding box coordinates to a text file
    txt_file = os.path.join(result_folder, image_name.split('.')[0] + '.txt')
    with open(txt_file, 'w') as f:
        if confidence is not None and bbox_coords is not None:
            f.write(f"{confidence}\n")
            f.write(f"{bbox_coords}\n")

print("Results saved in 'results-bbox' folder.")
