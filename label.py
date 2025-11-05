import os
from ultralytics import YOLO

# Configuration
images_folder = "augmented_frames"           # Folder with your images
labels_folder = "labels"           # Empty folder for labels
model_path = "best.pt"             # Your trained model

# Create labels folder if it doesn't exist
os.makedirs(labels_folder, exist_ok=True)

# Load your trained model
model = YOLO(model_path)

# Get all image files
image_files = [f for f in os.listdir(images_folder) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Processing {len(image_files)} images...")

for img_file in image_files:
    img_path = os.path.join(images_folder, img_file)
    
    # Create label filename with SAME name as image (just change extension to .txt)
    img_name_without_ext = os.path.splitext(img_file)[0]  # Remove extension
    label_file = img_name_without_ext + '.txt'            # Add .txt extension
    label_path = os.path.join(labels_folder, label_file)
    
    # Run detection
    results = model(img_path, verbose=False)
    
    # Write YOLO format labels
    with open(label_path, 'w') as f:
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get normalized coordinates
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only save if confidence > threshold
                    if confidence > 0.5:
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    print(f"Image: {img_file} → Label: {label_file}")

print("✅ Label generation complete!")