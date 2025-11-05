import os
from ultralytics import YOLO


model_path = "bestgithub.pt"  
input_video = "demovid.mp4"   
output_folder = "C:/Users/z00543ct/Desktop/helmet/output" 

HELMET_CLASS = 0              
CONFIDENCE_THRESHOLD = 0.5   
IoU_THRESHOLD = 0.7          


os.makedirs(output_folder, exist_ok=True)

print("Loading YOLO model...")

model = YOLO(model_path)

print(f"Starting helmet detection on: {input_video}")

results = model.predict(
    source=input_video,
    save=True,             
    save_txt=False,        
    project=output_folder, 
    name="helmet", 
    exist_ok=True,         
    classes=[HELMET_CLASS], 
    conf=CONFIDENCE_THRESHOLD,
    iou=IoU_THRESHOLD,     
    show_labels=True,       
    show_conf=True,        
    line_thickness=2,      
    imgsz=640              
)

output_path = os.path.join(output_folder, "helmet_detection")
print(f"Helmet detection complete!")
print(f"Results saved in: {output_path}")
print(f"Output video: {os.path.join(output_path, 'demovid.mp4')}")
